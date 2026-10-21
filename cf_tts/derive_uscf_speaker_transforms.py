#!/usr/bin/env python3
"""Derive open-set USCF speaker transforms from pre-extracted WavLM features.

For each LibriSpeech speaker, this implements the paper's least-squares
decoder construction using every available frame::

    C = X @ U
    S = pinv(C.T @ C) @ (C.T @ X)

``U`` is loaded through CF-TTS's shared ``load_content_projection`` helper,
so it is the same UTXSS 1024->75 projection used by CF-TTS.  The resulting
``S`` is a compatible 75->1024 target-speaker transform. Cross-products are
accumulated on one GPU in bounded batches; a speaker's full frame matrix is
never held in memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from linearvc.cf_tts.utils.common import load_content_projection


CONVERTER_VERSION = 1
PROVENANCE_NAME = "speaker-transform-provenance.json"
STATS_NAME = "speaker-transform-stats.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--max-batch-frames", type=int, default=0,
        help="Frames per GPU cross-product batch. 0 probes the largest fitting CUDA batch.",
    )
    parser.add_argument(
        "--max-host-batch-gib", type=float, default=96.0,
        help="Host-RAM cap for one batch of raw tensors (default: %(default)s GiB).",
    )
    parser.add_argument(
        "--progress-seconds", type=float, default=30.0,
        help="Emit a durable global throughput/ETA line at this cadence.",
    )
    parser.add_argument(
        "--overwrite-speakers", action="store_true",
        help="Recompute speaker matrices that already exist. The default resumes by skipping them.",
    )
    args = parser.parse_args()
    if args.max_batch_frames < 0:
        parser.error("--max-batch-frames must be non-negative")
    if args.max_host_batch_gib <= 0:
        parser.error("--max-host-batch-gib must be positive")
    if args.progress_seconds <= 0:
        parser.error("--progress-seconds must be positive")
    return args


def format_bytes(value: int | float) -> str:
    value = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


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


def load_tensor(path: Path) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected tensor in {path}, found {type(value).__name__}.")
    if value.ndim != 2 or not value.is_floating_point():
        raise ValueError(f"Expected rank-2 floating tensor in {path}, got {tuple(value.shape)} {value.dtype}.")
    return value.contiguous()


def save_numpy_atomically(value: np.ndarray | dict[str, np.ndarray], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, value)
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
    return hashlib.sha256(projection.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def speaker_id_for(path: Path) -> str:
    # LibriSpeech file names retain the <speaker>-<chapter>-<utterance> prefix,
    # even if the extractor preserved a suffix such as ".flac.pt".
    speaker_id = path.name.split("-", 1)[0]
    if not speaker_id.isdigit():
        raise ValueError(f"Could not infer LibriSpeech speaker ID from {path.name!r}.")
    return speaker_id


def load_projection(config_path: Path, device: torch.device) -> tuple[dict[str, Any], torch.Tensor]:
    config = yaml.safe_load(config_path.read_text())
    factorization = config["training"]["content_factorization"]
    projection = load_content_projection(factorization, device).contiguous()
    if projection.ndim != 2:
        raise ValueError(f"Expected rank-2 projection, got {tuple(projection.shape)}.")
    return factorization, projection


def make_provenance(
    input_root: Path,
    output_dir: Path,
    config_path: Path,
    factorization: dict[str, Any],
    projection: torch.Tensor,
) -> dict[str, Any]:
    return {
        "converter": "linearvc.cf_tts.derive_uscf_speaker_transforms",
        "converter_version": CONVERTER_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "config": str(config_path),
        "content_factorization": factorization,
        "projection_shape": list(projection.shape),
        "projection_dtype": str(projection.dtype),
        "projection_sha256": projection_sha256(projection),
        "speaker_transform_formula": "pinv((X @ U).T @ (X @ U)) @ ((X @ U).T @ X)",
        "speaker_transform_dtype": "float32",
        "uses_all_available_frames": True,
    }


def ensure_provenance(provenance: dict[str, Any], path: Path) -> None:
    if not path.is_file():
        save_json_atomically(provenance, path)
        return
    existing = json.loads(path.read_text())
    keys = (
        "converter", "converter_version", "input_root", "output_dir", "config",
        "content_factorization", "projection_shape", "projection_dtype",
        "projection_sha256", "speaker_transform_formula", "speaker_transform_dtype",
        "uses_all_available_frames",
    )
    differences = {key: (existing.get(key), provenance.get(key)) for key in keys if existing.get(key) != provenance.get(key)}
    if differences:
        raise RuntimeError(f"Existing provenance does not match requested derivation: {differences}")


def probe_fits(projection: torch.Tensor, frames: int, device: torch.device) -> bool:
    source = content = source64 = content64 = None
    try:
        source = torch.empty((frames, projection.shape[0]), dtype=torch.float32, device=device)
        content = torch.empty((frames, projection.shape[1]), dtype=torch.float32, device=device)
        # The least-squares sufficient statistics use float64 accumulation.
        source64 = torch.empty((frames, projection.shape[0]), dtype=torch.float64, device=device)
        content64 = torch.empty((frames, projection.shape[1]), dtype=torch.float64, device=device)
        probe_frames = min(frames, 4096)
        torch.mm(source[:probe_frames], projection, out=content[:probe_frames])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return True
    except torch.cuda.OutOfMemoryError:
        return False
    finally:
        del source, content, source64, content64
        if device.type == "cuda":
            torch.cuda.empty_cache()


def probe_batch_frames(projection: torch.Tensor, device: torch.device) -> int:
    free_bytes, _ = torch.cuda.mem_get_info(device)
    bytes_per_frame = 12 * sum(projection.shape)
    low, high, best = 1, max(1, free_bytes // bytes_per_frame), 0
    while low <= high:
        candidate = (low + high) // 2
        if probe_fits(projection, candidate, device):
            best = candidate
            low = candidate + 1
        else:
            high = candidate - 1
    if not best:
        raise RuntimeError("Unable to allocate one raw/content feature frame on the selected GPU.")
    return best


def accumulate_batch(
    tensors: list[torch.Tensor], projection: torch.Tensor, gram: torch.Tensor,
    cross: torch.Tensor, device: torch.device,
) -> int:
    source = content = source64 = content64 = None
    try:
        source = torch.cat(tensors, dim=0).to(device=device, dtype=torch.float32, non_blocking=False)
        content = torch.mm(source, projection)
        # The content projection itself is float32, exactly as in CF-TTS. Use
        # float64 only for least-squares sufficient statistics so summing all
        # frames stays stable across batch boundaries.
        source64 = source.to(dtype=torch.float64)
        content64 = content.to(dtype=torch.float64)
        gram.add_(content64.T @ content64)
        cross.add_(content64.T @ source64)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return int(source.shape[0])
    finally:
        del source, content, source64, content64
        if device.type == "cuda":
            torch.cuda.empty_cache()


def emit_progress(
    started: float, total_bytes: int, completed_bytes: int, total_files: int,
    completed_files: int, total_speakers: int, completed_speakers: int,
    completed_frames: int,
) -> None:
    elapsed = max(time.monotonic() - started, 1e-9)
    rate = completed_bytes / elapsed
    eta = (total_bytes - completed_bytes) / rate if rate else math.inf
    percent = 100.0 * completed_bytes / total_bytes if total_bytes else 100.0
    print(
        "[progress] "
        f"speakers={completed_speakers}/{total_speakers} files={completed_files}/{total_files} "
        f"input={format_bytes(completed_bytes)}/{format_bytes(total_bytes)} ({percent:.1f}%) "
        f"frames={completed_frames:,} throughput={format_bytes(rate)}/s ETA={format_duration(eta)}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    if not input_root.is_dir():
        raise SystemExit(f"Input root does not exist: {input_root}")
    if not config_path.is_file():
        raise SystemExit(f"CF-TTS config does not exist: {config_path}")
    if output_dir == input_root or input_root in output_dir.parents:
        raise SystemExit("Output directory must be outside the raw feature tree.")

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device requests CUDA but no CUDA GPU is available.")
        torch.cuda.set_device(device)
    elif args.max_batch_frames == 0:
        raise SystemExit("A CPU test run requires an explicit --max-batch-frames value.")

    factorization, projection = load_projection(config_path, device)
    source_dim, content_dim = projection.shape
    provenance = make_provenance(input_root, output_dir, config_path, factorization, projection)
    ensure_provenance(provenance, output_dir / PROVENANCE_NAME)

    grouped_paths: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(input_root.rglob("*.pt")):
        if path.is_file():
            grouped_paths[speaker_id_for(path)].append(path)
    if not grouped_paths:
        raise SystemExit(f"No .pt tensors found below {input_root}")

    if args.max_batch_frames:
        gpu_batch_frames, probe_mode = args.max_batch_frames, "explicit"
    elif device.type == "cuda":
        print("Probing the largest raw/content frame batch that fits the allocated GPU...", flush=True)
        gpu_batch_frames, probe_mode = probe_batch_frames(projection, device), "auto-probed"
    else:
        raise AssertionError("CPU runs require an explicit batch size above.")
    # A batch briefly retains individual tensors and a concatenation. Bound this
    # by two float32 source copies; GPU capacity remains the primary limit.
    host_batch_frames = max(1, int(args.max_host_batch_gib * 1024**3) // (8 * source_dim))
    max_batch_frames = min(gpu_batch_frames, host_batch_frames)

    speaker_paths = {speaker: output_dir / "speakers" / f"{speaker}.npy" for speaker in grouped_paths}
    pending_speakers = [speaker for speaker in sorted(grouped_paths) if args.overwrite_speakers or not speaker_paths[speaker].is_file()]
    pending_files = [path for speaker in pending_speakers for path in grouped_paths[speaker]]
    source_bytes = {path: path.stat().st_size for path in pending_files}
    total_bytes, total_files = sum(source_bytes.values()), len(pending_files)

    print(
        f"Projection: {source_dim}->{content_dim}; mode={factorization.get('content_projection', 'utxss')}; "
        f"sha256={provenance['projection_sha256']}", flush=True,
    )
    print(
        f"Batch capacity: gpu={gpu_batch_frames:,} frames ({probe_mode}), host={host_batch_frames:,} frames, "
        f"using={max_batch_frames:,} frames.", flush=True,
    )
    print(
        f"Discovered {len(grouped_paths):,} speakers / {sum(map(len, grouped_paths.values())):,} tensors; "
        f"{len(pending_speakers):,} speakers require derivation.", flush=True,
    )

    started = last_report = time.monotonic()
    completed_bytes = completed_files = completed_frames = completed_speakers = 0
    stats_path = output_dir / STATS_NAME
    stats: dict[str, dict[str, Any]] = json.loads(stats_path.read_text()) if stats_path.is_file() and not args.overwrite_speakers else {}

    for speaker in pending_speakers:
        paths = grouped_paths[speaker]
        print(f"[speaker] {speaker}: {len(paths):,} tensors; deriving from every available frame.", flush=True)
        gram = torch.zeros((content_dim, content_dim), dtype=torch.float64, device=device)
        cross = torch.zeros((content_dim, source_dim), dtype=torch.float64, device=device)
        batch_tensors: list[torch.Tensor] = []
        batch_frames = batch_bytes = batch_files = speaker_frames = speaker_files = 0

        def flush_batch() -> None:
            nonlocal batch_tensors, batch_frames, batch_bytes, batch_files
            nonlocal speaker_frames, completed_bytes, completed_files
            if batch_tensors:
                speaker_frames += accumulate_batch(batch_tensors, projection, gram, cross, device)
                completed_bytes += batch_bytes
                completed_files += batch_files
                batch_tensors, batch_frames, batch_bytes, batch_files = [], 0, 0, 0

        for path in paths:
            tensor = load_tensor(path)
            if tensor.shape[1] != source_dim:
                raise ValueError(f"Feature width mismatch in {path}: got {tensor.shape[1]}, expected {source_dim}.")
            if tensor.shape[0] > max_batch_frames:
                raise RuntimeError(
                    f"Single tensor exceeds batch capacity: {path} has {tensor.shape[0]:,} frames, "
                    f"capacity is {max_batch_frames:,}."
                )
            if batch_tensors and batch_frames + tensor.shape[0] > max_batch_frames:
                flush_batch()
            batch_tensors.append(tensor)
            batch_frames += tensor.shape[0]
            batch_bytes += source_bytes[path]
            batch_files += 1
            speaker_files += 1
            if time.monotonic() - last_report >= args.progress_seconds:
                emit_progress(
                    started, total_bytes, completed_bytes, total_files, completed_files,
                    len(pending_speakers), completed_speakers, completed_frames + speaker_frames,
                )
                last_report = time.monotonic()
        flush_batch()

        transform = torch.linalg.pinv(gram) @ cross
        stats[speaker] = {
            "files": speaker_files,
            "frames": speaker_frames,
            "gram_rank": int(torch.linalg.matrix_rank(gram).item()),
            "gram_condition_number": float(torch.linalg.cond(gram).item()),
            "transform_path": str(speaker_paths[speaker]),
        }
        save_numpy_atomically(transform.to(dtype=torch.float32).cpu().numpy(), speaker_paths[speaker])
        save_json_atomically(stats, stats_path)
        completed_speakers += 1
        completed_frames += speaker_frames
        emit_progress(
            started, total_bytes, completed_bytes, total_files, completed_files,
            len(pending_speakers), completed_speakers, completed_frames,
        )

    transforms: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for speaker in sorted(grouped_paths):
        path = speaker_paths[speaker]
        if path.is_file():
            transforms[speaker] = np.load(path)
        else:
            missing.append(speaker)
    if missing:
        raise RuntimeError(f"Cannot build transforms.npy; missing speaker matrices: {', '.join(missing[:10])}")
    save_numpy_atomically(transforms, output_dir / "transforms.npy")
    print(
        f"Completed {len(transforms):,} speaker transforms. "
        f"CF-TTS-compatible dictionary: {output_dir / 'transforms.npy'}", flush=True,
    )


if __name__ == "__main__":
    main()
