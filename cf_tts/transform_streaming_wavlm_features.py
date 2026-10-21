#!/usr/bin/env python3
"""Project streamed WavLM tensors into the CF-TTS content space.

The projection is deliberately loaded through ``load_content_projection`` so
this job uses precisely the same UTXSS (or explicitly requested legacy
``pinv_anchor``) construction as CF-TTS training and inference.  Input tensor
paths are mirrored below the output root and each completed output is written
atomically, making interrupted Slurm jobs safely resumable with
``--skip-existing``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from linearvc.cf_tts.utils.common import load_content_projection


MANIFEST_NAME = "conversion-provenance.json"
CONVERTER_VERSION = 1


@dataclass
class FeatureItem:
    source: Path
    destination: Path
    source_bytes: int
    tensor: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    projection_source = parser.add_mutually_exclusive_group(required=True)
    projection_source.add_argument(
        "--config",
        type=Path,
        help="CF-TTS YAML config whose content_factorization section defines the projection.",
    )
    projection_source.add_argument(
        "--projection-path",
        type=Path,
        help="Explicit 2-D .npy projection matrix, for example a completed UTXSS.npy.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dtype",
        choices=("float16", "float32"),
        default="float32",
        help="Storage dtype. float32 preserves the same matmul result used in CF-TTS.",
    )
    parser.add_argument(
        "--max-batch-frames",
        type=int,
        default=0,
        help="Frames per GPU matmul. 0 probes the largest fitting CUDA batch.",
    )
    parser.add_argument(
        "--max-host-batch-gib",
        type=float,
        default=96.0,
        help="Estimated peak host-RAM cap for one conversion batch (default: %(default)s GiB).",
    )
    parser.add_argument(
        "--progress-seconds",
        type=float,
        default=30.0,
        help="Emit a durable throughput/ETA progress line at this cadence.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Allow an existing destination file to be atomically replaced."
        " The default refuses to overwrite converted tensors.",
    )
    args = parser.parse_args()
    if args.max_batch_frames < 0:
        parser.error("--max-batch-frames must be non-negative")
    if args.max_host_batch_gib <= 0:
        parser.error("--max-host-batch-gib must be positive")
    if args.progress_seconds <= 0:
        parser.error("--progress-seconds must be positive")
    return args


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds = int(round(seconds))
    days, seconds = divmod(seconds, 24 * 60 * 60)
    hours, seconds = divmod(seconds, 60 * 60)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_bytes(num_bytes: int | float) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def torch_load_tensor(path: Path) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected a tensor in {path}, found {type(value).__name__}.")
    if value.ndim != 2:
        raise ValueError(f"Expected rank-2 [frames, features] tensor in {path}, got {tuple(value.shape)}.")
    if not value.is_floating_point():
        raise TypeError(f"Expected floating-point tensor in {path}, got {value.dtype}.")
    return value.contiguous()


def destination_for(source: Path, input_root: Path, output_root: Path) -> Path:
    return output_root / source.relative_to(input_root)


def save_tensor_atomically(tensor: torch.Tensor, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        torch.save(tensor, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def save_json_atomically(value: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def projection_sha256(projection: torch.Tensor) -> str:
    array = projection.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def conversion_provenance(
    *,
    input_root: Path,
    output_root: Path,
    config_path: Path | None,
    factorization: dict[str, Any],
    projection: torch.Tensor,
    output_dtype: torch.dtype,
) -> dict[str, Any]:
    return {
        "converter": "linearvc.cf_tts.transform_streaming_wavlm_features",
        "converter_version": CONVERTER_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "config": None if config_path is None else str(config_path),
        "content_factorization": factorization,
        "projection_shape": list(projection.shape),
        "projection_dtype": str(projection.dtype),
        "projection_sha256": projection_sha256(projection),
        "operation": "torch.matmul(features.float(), projection.float())",
        "output_dtype": str(output_dtype),
    }


def ensure_compatible_provenance(provenance: dict[str, Any], destination: Path) -> None:
    if not destination.is_file():
        save_json_atomically(provenance, destination)
        return
    existing = json.loads(destination.read_text())
    comparison_keys = (
        "converter",
        "converter_version",
        "input_root",
        "output_root",
        "config",
        "content_factorization",
        "projection_shape",
        "projection_dtype",
        "projection_sha256",
        "operation",
        "output_dtype",
    )
    mismatch = {
        key: {"existing": existing.get(key), "requested": provenance.get(key)}
        for key in comparison_keys
        if existing.get(key) != provenance.get(key)
    }
    if mismatch:
        raise RuntimeError(
            f"Existing conversion provenance at {destination} does not match this run: "
            f"{json.dumps(mismatch, sort_keys=True)}"
        )


def probe_fits(
    projection: torch.Tensor,
    frame_count: int,
    device: torch.device,
) -> bool:
    """Check allocation capacity without performing an expensive full probe GEMM."""
    source = output = None
    try:
        source = torch.empty((frame_count, projection.shape[0]), dtype=torch.float32, device=device)
        output = torch.empty((frame_count, projection.shape[1]), dtype=torch.float32, device=device)
        # Exercise the same GEMM code path on a small slice while retaining the
        # complete source/output allocations that determine capacity.
        probe_frames = min(frame_count, 4096)
        torch.mm(source[:probe_frames], projection, out=output[:probe_frames])
        torch.cuda.synchronize(device)
        return True
    except torch.cuda.OutOfMemoryError:
        return False
    finally:
        del source, output
        torch.cuda.empty_cache()


def probe_largest_batch_frames(projection: torch.Tensor, device: torch.device) -> int:
    free_bytes, _ = torch.cuda.mem_get_info(device)
    bytes_per_frame = 4 * (projection.shape[0] + projection.shape[1])
    upper = max(1, free_bytes // bytes_per_frame)
    lower = 1
    best = 0
    while lower <= upper:
        candidate = (lower + upper) // 2
        if probe_fits(projection, candidate, device):
            best = candidate
            lower = candidate + 1
        else:
            upper = candidate - 1
    if best == 0:
        raise RuntimeError("Unable to allocate even one input/output feature frame on the selected GPU.")
    return best


def convert_batch(
    items: Sequence[FeatureItem],
    projection: torch.Tensor,
    device: torch.device,
    output_dtype: torch.dtype,
    allow_existing_output: bool,
) -> list[tuple[FeatureItem, torch.Tensor]]:
    source = gpu_source = gpu_output = output = None
    try:
        source = torch.cat([item.tensor for item in items], dim=0)
        gpu_source = source.to(device=device, dtype=torch.float32, non_blocking=False)
        gpu_output = torch.empty(
            (gpu_source.shape[0], projection.shape[1]), dtype=torch.float32, device=device
        )
        torch.mm(gpu_source, projection, out=gpu_output)
        output = gpu_output.to(device="cpu", dtype=output_dtype, non_blocking=False)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        converted: list[tuple[FeatureItem, torch.Tensor]] = []
        start = 0
        for item in items:
            end = start + item.tensor.shape[0]
            if item.destination.exists() and not allow_existing_output:
                raise FileExistsError(
                    f"Refusing to overwrite existing converted feature: {item.destination}. "
                    "Use --skip-existing to resume or --allow-existing-output to replace it."
                )
            converted.append((item, output[start:end].clone()))
            start = end
        return converted
    finally:
        del source, gpu_source, gpu_output, output
        if device.type == "cuda":
            torch.cuda.empty_cache()


def load_config_projection(config_path: Path, device: torch.device) -> tuple[dict[str, Any], torch.Tensor]:
    config = yaml.safe_load(config_path.read_text())
    factorization = config["training"]["content_factorization"]
    projection = load_content_projection(factorization, device)
    if projection.ndim != 2:
        raise ValueError(f"CF-TTS projection must be rank 2, got {tuple(projection.shape)}.")
    return factorization, projection.contiguous()


def load_explicit_projection(projection_path: Path, device: torch.device) -> tuple[dict[str, Any], torch.Tensor]:
    """Load a directly specified content projection with self-describing provenance."""
    projection_path = projection_path.expanduser().resolve()
    if not projection_path.is_file():
        raise FileNotFoundError(f"Explicit projection does not exist: {projection_path}")
    projection = np.load(projection_path)
    if projection.ndim != 2:
        raise ValueError(f"Explicit projection must be rank 2, got {projection.shape} from {projection_path}.")
    factorization = {
        "type": "content",
        "content_projection": "utxss",
        "content_projection_file": str(projection_path),
        "projection_source": "explicit_path",
    }
    return factorization, torch.as_tensor(projection, dtype=torch.float32, device=device).contiguous()


def report_progress(
    *,
    start_time: float,
    total_bytes: int,
    completed_bytes: int,
    total_files: int,
    completed_files: int,
    completed_frames: int,
    batch_count: int,
    max_batch_frames: int,
) -> None:
    elapsed = max(time.monotonic() - start_time, 1e-9)
    bytes_per_second = completed_bytes / elapsed
    remaining_seconds = (total_bytes - completed_bytes) / bytes_per_second if bytes_per_second else math.inf
    percent = 100.0 * completed_bytes / total_bytes if total_bytes else 100.0
    print(
        "[progress] "
        f"files={completed_files}/{total_files} "
        f"input={format_bytes(completed_bytes)}/{format_bytes(total_bytes)} ({percent:.1f}%) "
        f"frames={completed_frames:,} batches={batch_count:,} "
        f"throughput={format_bytes(bytes_per_second)}/s "
        f"ETA={format_duration(remaining_seconds)} "
        f"max_batch_frames={max_batch_frames:,}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    config_path = None if args.config is None else args.config.expanduser().resolve()
    projection_path = None if args.projection_path is None else args.projection_path.expanduser().resolve()
    if not input_root.is_dir():
        raise SystemExit(f"Input root does not exist: {input_root}")
    if config_path is not None and not config_path.is_file():
        raise SystemExit(f"CF-TTS config does not exist: {config_path}")
    if projection_path is not None and not projection_path.is_file():
        raise SystemExit(f"Explicit projection does not exist: {projection_path}")
    if output_root == input_root or input_root in output_root.parents:
        raise SystemExit("Output root must be outside the input feature tree.")

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device requests CUDA but no CUDA device is available.")
        torch.cuda.set_device(device)
    elif args.max_batch_frames == 0:
        raise SystemExit("A CPU run requires an explicit --max-batch-frames value; GPU runs auto-probe by default.")

    if config_path is not None:
        factorization, projection = load_config_projection(config_path, device)
    else:
        assert projection_path is not None
        factorization, projection = load_explicit_projection(projection_path, device)
    output_dtype = torch.float16 if args.output_dtype == "float16" else torch.float32
    provenance = conversion_provenance(
        input_root=input_root,
        output_root=output_root,
        config_path=config_path,
        factorization=factorization,
        projection=projection,
        output_dtype=output_dtype,
    )
    ensure_compatible_provenance(provenance, output_root / MANIFEST_NAME)

    source_paths = sorted(path for path in input_root.rglob("*.pt") if path.is_file())
    if not source_paths:
        raise SystemExit(f"No .pt tensors found below {input_root}")
    remaining_paths: list[Path] = []
    skipped = 0
    for source in source_paths:
        destination = destination_for(source, input_root, output_root)
        if destination.is_file() and args.skip_existing:
            skipped += 1
        else:
            remaining_paths.append(source)

    source_dim, output_dim = projection.shape
    if args.max_batch_frames:
        gpu_batch_frames = args.max_batch_frames
        probe_label = "explicit"
    elif device.type == "cuda":
        print("Probing the largest GPU input/output frame batch...", flush=True)
        gpu_batch_frames = probe_largest_batch_frames(projection, device)
        probe_label = "auto-probed"
    else:
        raise AssertionError("CPU handling above requires an explicit batch size.")

    # ``items`` retains each loaded source tensor while ``torch.cat`` creates a
    # second contiguous source buffer.  The receive buffer and per-file clones
    # likewise coexist until all outputs have been written.  Budget both copies
    # of source and output so max-host-batch-gib is an actual *peak* RAM cap,
    # rather than merely the size of one logical input/output batch.
    host_bytes_per_frame = 2 * 4 * (source_dim + output_dim)
    host_batch_frames = max(1, int(args.max_host_batch_gib * 1024**3) // host_bytes_per_frame)
    max_batch_frames = min(gpu_batch_frames, host_batch_frames)
    print(
        f"Projection: {source_dim} -> {output_dim}; mode={factorization.get('content_projection', 'utxss')}; "
        f"sha256={provenance['projection_sha256']}",
        flush=True,
    )
    print(
        f"Batch capacity: gpu={gpu_batch_frames:,} frames ({probe_label}), "
        f"host={host_batch_frames:,} frames, using={max_batch_frames:,} frames.",
        flush=True,
    )
    print(
        f"Discovered {len(source_paths):,} source tensors; {skipped:,} already converted; "
        f"{len(remaining_paths):,} require conversion.",
        flush=True,
    )
    if not remaining_paths:
        print("Nothing to convert; provenance matches and every destination exists.", flush=True)
        return

    source_sizes = {path: path.stat().st_size for path in remaining_paths}
    total_bytes = sum(source_sizes.values())
    completed_bytes = completed_files = completed_frames = batch_count = 0
    started = time.monotonic()
    last_report = started
    carry: FeatureItem | None = None
    path_index = 0

    while carry is not None or path_index < len(remaining_paths):
        items: list[FeatureItem] = []
        frame_count = 0
        while carry is not None or path_index < len(remaining_paths):
            if carry is None:
                source = remaining_paths[path_index]
                path_index += 1
                tensor = torch_load_tensor(source)
                if tensor.shape[1] != source_dim:
                    raise ValueError(
                        f"Feature width mismatch in {source}: got {tensor.shape[1]}, expected {source_dim}."
                    )
                carry = FeatureItem(
                    source=source,
                    destination=destination_for(source, input_root, output_root),
                    source_bytes=source_sizes[source],
                    tensor=tensor,
                )
            if carry.tensor.shape[0] > max_batch_frames:
                raise RuntimeError(
                    f"One tensor exceeds the probed frame capacity: {carry.source} has "
                    f"{carry.tensor.shape[0]:,} frames but the maximum batch is {max_batch_frames:,}. "
                    "Re-run with a larger GPU or add chunked-single-tensor support before conversion."
                )
            if items and frame_count + carry.tensor.shape[0] > max_batch_frames:
                break
            items.append(carry)
            frame_count += carry.tensor.shape[0]
            carry = None

        try:
            converted = convert_batch(
                items,
                projection,
                device,
                output_dtype,
                args.allow_existing_output,
            )
        except torch.cuda.OutOfMemoryError:
            if len(items) == 1:
                raise
            # The allocation probe leaves no deliberate safety margin.  If a
            # real GEMM needs extra workspace, retry with a smaller exact batch
            # and retain that discovered limit for the rest of this run.
            max_batch_frames = max(1, frame_count // 2)
            # The current batch has not written any output. Rewind its source
            # paths (and a carried-over item, if present) for a clean retry.
            path_index -= len(items)
            if carry is not None:
                path_index -= 1
                carry = None
            print(
                f"CUDA OOM at {frame_count:,} frames; retrying with max_batch_frames={max_batch_frames:,}.",
                flush=True,
            )
            continue

        for item, tensor in converted:
            save_tensor_atomically(tensor, item.destination)
            completed_bytes += item.source_bytes
            completed_files += 1
            completed_frames += item.tensor.shape[0]
        batch_count += 1
        now = time.monotonic()
        if now - last_report >= args.progress_seconds:
            report_progress(
                start_time=started,
                total_bytes=total_bytes,
                completed_bytes=completed_bytes,
                total_files=len(remaining_paths),
                completed_files=completed_files,
                completed_frames=completed_frames,
                batch_count=batch_count,
                max_batch_frames=max_batch_frames,
            )
            last_report = now

    report_progress(
        start_time=started,
        total_bytes=total_bytes,
        completed_bytes=completed_bytes,
        total_files=len(remaining_paths),
        completed_files=completed_files,
        completed_frames=completed_frames,
        batch_count=batch_count,
        max_batch_frames=max_batch_frames,
    )
    print("Conversion completed successfully.", flush=True)


if __name__ == "__main__":
    main()
