#!/usr/bin/env python3
"""Build a legacy-compatible LibriSpeech SVD factorization and UTXSS basis.

The input is a recursive folder of extracted per-utterance WavLM ``.pt``
tensors.  LibriSpeech speaker IDs are read from the usual
``<speaker>-<chapter>-<utterance>.pt`` filename.  All utterances for the
selected source speaker are concatenated, then evenly subsampled if needed.
All utterances for every speaker are likewise concatenated while that speaker
is aligned.  Each source frame is cosine-nearest-neighbour aligned to that
speaker's frames.  The concatenated alignment matrix is factorized as::

    XS = U @ diag(S) @ VT

The speaker transforms are the speaker blocks of ``VT``.  The global content
projection is then derived exactly as in the legacy LinearVC implementation::

    UTXS = lstsq(XS.reshape(-1, feature_dim), repeat(U, speakers))
    UTXSS = UTXS @ diag(S)

The result is the familiar 1024->rank ``UTXSS.npy`` alongside ``U.npy``,
``S.npy``, ``VT.npy``, and ``transforms.npy``. ``XS.npy`` can optionally be
retained for legacy workflows, but launchers normally place it in job-local
temporary storage and discard it after the final artifacts are written. The
expensive work belongs in a Slurm allocation; this module intentionally does
not create one itself.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.extmath import randomized_svd
import torch
import torch.nn.functional as F


FINAL_ARTIFACTS = (
    "XS.npy",
    "U.npy",
    "S.npy",
    "VT.npy",
    "transforms.npy",
    "UTXSS.npy",
    "factorization-provenance.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-root",
        type=Path,
        required=True,
        help="Recursive root of extracted per-utterance WavLM .pt tensors.",
    )
    parser.add_argument("--source-speaker", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument(
        "--xs-work-dir",
        type=Path,
        help=(
            "Directory for the large in-progress XS memmap. Defaults to --output-dir "
            "for legacy behavior. Pair with --discard-xs-after-factorization to keep XS transient."
        ),
    )
    parser.add_argument(
        "--discard-xs-after-factorization",
        action="store_true",
        help="Delete generated XS after SVD/UTXSS succeeds instead of publishing XS.npy to --output-dir.",
    )
    parser.add_argument(
        "--existing-xs",
        type=Path,
        help=(
            "Resume from a completed XS .npy matrix. Skips feature loading and alignment; "
            "the matrix is opened read-only and final artifacts are written to --output-dir."
        ),
    )
    parser.add_argument(
        "--max-source-frames",
        type=int,
        default=32768,
        help="Use at most this many evenly spaced source frames; 0 keeps all source frames.",
    )
    parser.add_argument(
        "--speaker-id-regex",
        default=r"^([0-9]+)-",
        help="Regex applied to each .pt filename stem; capture group 1 is its speaker ID.",
    )
    parser.add_argument("--svd-n-iter", type=int, default=7)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--alignment-device",
        default="cpu",
        help="Device for cosine nearest-neighbour alignment, e.g. cpu or cuda:0.",
    )
    parser.add_argument(
        "--alignment-backend",
        choices=("auto", "torch", "cuvs"),
        default="auto",
        help=(
            "GPU nearest-neighbour backend. 'auto' uses cuVS brute force when "
            "available and otherwise PyTorch; CPU alignment always uses scikit-learn."
        ),
    )
    parser.add_argument(
        "--alignment-batch-frames",
        type=int,
        default=0,
        help="Source frames per GPU cosine-search batch; 0 chooses a memory-derived batch size.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="CPU workers used by cosine nearest-neighbour alignment.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the known output artifacts in --output-dir.",
    )
    args = parser.parse_args()
    if args.rank < 1:
        parser.error("--rank must be positive")
    if args.max_source_frames < 0:
        parser.error("--max-source-frames must be non-negative")
    if args.svd_n_iter < 0:
        parser.error("--svd-n-iter must be non-negative")
    if args.n_jobs < 1:
        parser.error("--n-jobs must be positive")
    if args.alignment_batch_frames < 0:
        parser.error("--alignment-batch-frames must be non-negative")
    try:
        pattern = re.compile(args.speaker_id_regex)
    except re.error as exc:
        parser.error(f"Invalid --speaker-id-regex: {exc}")
    if pattern.groups < 1:
        parser.error("--speaker-id-regex must contain a capture group for the speaker ID")
    return args


def atomic_save_npy(value: object, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, value)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_save_json(value: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_utterance_tensor(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Feature tensor does not exist: {path}")
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor in {path}, found {type(value).__name__}")
    value = value.detach().cpu().numpy()
    if value.ndim != 2:
        raise ValueError(f"Feature tensor must be rank 2, got {value.shape} at {path}")
    if value.shape[0] < 1 or value.shape[1] < 1:
        raise ValueError(f"Feature tensor must be non-empty, got {value.shape} at {path}")
    if not np.issubdtype(value.dtype, np.floating):
        raise ValueError(f"Feature tensor must be floating point, got {value.dtype} at {path}")
    return np.asarray(value, dtype=np.float32)


def discover_speaker_utterances(root: Path, speaker_id_regex: str) -> dict[str, list[Path]]:
    if not root.is_dir():
        raise NotADirectoryError(f"Feature root does not exist: {root}")
    paths = sorted(path for path in root.rglob("*.pt") if path.is_file())
    if not paths:
        raise RuntimeError(f"No extracted .pt feature tensors found under {root}")
    pattern = re.compile(speaker_id_regex)
    grouped: dict[str, list[Path]] = {}
    unmatched = []
    for path in paths:
        match = pattern.search(path.stem)
        if match is None:
            unmatched.append(path)
            continue
        grouped.setdefault(match.group(1), []).append(path)
    if unmatched:
        examples = ", ".join(str(path) for path in unmatched[:3])
        raise ValueError(
            f"Could not infer a speaker ID from {len(unmatched)} .pt filename(s) using "
            f"{speaker_id_regex!r}; examples: {examples}"
        )
    return {speaker: grouped[speaker] for speaker in sorted(grouped)}


def load_speaker_matrix(speaker_id: str, paths: Sequence[Path]) -> np.ndarray:
    matrices = [load_utterance_tensor(path) for path in paths]
    feature_dims = {matrix.shape[1] for matrix in matrices}
    if len(feature_dims) != 1:
        raise ValueError(
            f"Speaker {speaker_id!r} has inconsistent feature dimensions: {sorted(feature_dims)}"
        )
    return np.concatenate(matrices, axis=0)


def select_source_frames(source: np.ndarray, maximum: int) -> np.ndarray:
    if maximum == 0 or source.shape[0] <= maximum:
        return np.asarray(source, dtype=np.float32)
    indices = np.linspace(0, source.shape[0] - 1, num=maximum, dtype=np.int64)
    return np.asarray(source[indices], dtype=np.float32)


def cpu_nearest_indices(source: np.ndarray, references: np.ndarray, n_jobs: int) -> np.ndarray:
    neighbors = NearestNeighbors(n_neighbors=1, metric="cosine", n_jobs=n_jobs)
    neighbors.fit(references)
    return neighbors.kneighbors(source, return_distance=False).reshape(-1)


def gpu_batch_frames(
    source_frames: int,
    reference_frames: int,
    feature_dim: int,
    device: torch.device,
    requested: int,
) -> int:
    """Choose a large score-matrix batch without materializing all pairwise scores."""
    if requested:
        return min(requested, source_frames)
    free_bytes, _ = torch.cuda.mem_get_info(device)
    # One float32 score per source/reference pair dominates. Source, normalized
    # source, and allocator overhead are represented by the feature-dimension
    # term. Leave 10% free for kernels and the already resident reference matrix.
    bytes_per_source_frame = 4 * (reference_frames + 3 * feature_dim)
    return max(1, min(source_frames, int((free_bytes * 0.90) // bytes_per_source_frame)))


def gpu_nearest_indices(
    source: np.ndarray,
    references: np.ndarray,
    device: torch.device,
    requested_batch_frames: int,
    speaker_id: str,
) -> np.ndarray:
    """Find cosine nearest neighbours with batched GPU matrix multiplication.

    ``scores`` is never larger than ``[batch_frames, reference_frames]``.  If
    a memory estimate is optimistic on a particular GPU, retry that batch at
    half size instead of failing the full factorization.
    """
    reference_tensor = torch.from_numpy(np.ascontiguousarray(references)).to(device, dtype=torch.float32)
    reference_t = F.normalize(reference_tensor, dim=1).transpose(0, 1).contiguous()
    del reference_tensor
    batch_frames = gpu_batch_frames(
        source.shape[0], references.shape[0], source.shape[1], device, requested_batch_frames
    )
    print(
        f"GPU cosine alignment for speaker {speaker_id}: device={device}, "
        f"batch_frames={batch_frames:,}, reference_frames={references.shape[0]:,}",
        flush=True,
    )
    indices = np.empty(source.shape[0], dtype=np.int64)
    offset = 0
    with torch.inference_mode():
        while offset < source.shape[0]:
            current_batch = min(batch_frames, source.shape[0] - offset)
            try:
                source_tensor = torch.from_numpy(
                    np.ascontiguousarray(source[offset:offset + current_batch])
                ).to(device, dtype=torch.float32)
                scores = torch.mm(F.normalize(source_tensor, dim=1), reference_t)
                indices[offset:offset + current_batch] = scores.argmax(dim=1).cpu().numpy()
                offset += current_batch
                del source_tensor, scores
            except torch.OutOfMemoryError:
                if current_batch == 1:
                    raise RuntimeError(
                        f"GPU alignment cannot fit one source frame against speaker {speaker_id!r} "
                        f"({references.shape[0]:,} reference frames) on {device}."
                    ) from None
                batch_frames = max(1, current_batch // 2)
                torch.cuda.empty_cache()
                print(
                    f"GPU alignment OOM for speaker {speaker_id}; retrying at "
                    f"batch_frames={batch_frames:,}",
                    flush=True,
                )
    del reference_t
    torch.cuda.empty_cache()
    return indices


def load_cuvs() -> tuple[Any, Any]:
    """Load the optional cuVS/CuPy brute-force backend with an actionable error."""
    try:
        import cupy as cp
        from cuvs.neighbors import brute_force
    except ImportError as exc:
        raise RuntimeError(
            "cuVS alignment was requested, but its optional dependencies are not installed. "
            "Run slurm/install_cuvs.sbatch in a CPU Slurm allocation, then retry."
        ) from exc
    return cp, brute_force


def cuvs_nearest_indices(
    source: np.ndarray,
    references: np.ndarray,
    device: torch.device,
    requested_batch_frames: int,
    speaker_id: str,
    cp: Any,
    brute_force: Any,
) -> np.ndarray:
    """Find exact cosine nearest neighbours through cuVS brute-force search.

    cuVS keeps its search workspace internal instead of materializing the full
    score matrix in Python.  It is still an exact brute-force search: no IVF,
    CAGRA, or other approximate index is used here.
    """
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    batch_frames = gpu_batch_frames(
        source.shape[0], references.shape[0], source.shape[1], device, requested_batch_frames
    )
    print(
        f"cuVS exact cosine alignment for speaker {speaker_id}: device=cuda:{device_index}, "
        f"batch_frames={batch_frames:,}, reference_frames={references.shape[0]:,}",
        flush=True,
    )
    indices = np.empty(source.shape[0], dtype=np.int64)
    with cp.cuda.Device(device_index):
        reference_device = cp.asarray(np.ascontiguousarray(references), dtype=cp.float32)
        index = brute_force.build(reference_device, metric="cosine")
        offset = 0
        try:
            while offset < source.shape[0]:
                current_batch = min(batch_frames, source.shape[0] - offset)
                try:
                    queries = cp.asarray(
                        np.ascontiguousarray(source[offset:offset + current_batch]),
                        dtype=cp.float32,
                    )
                    _, neighbor_indices = brute_force.search(index, queries, k=1)
                    indices[offset:offset + current_batch] = cp.asnumpy(neighbor_indices).reshape(-1)
                    offset += current_batch
                    del queries, neighbor_indices
                except cp.cuda.memory.OutOfMemoryError:
                    if current_batch == 1:
                        raise RuntimeError(
                            f"cuVS alignment cannot fit one source frame against speaker {speaker_id!r} "
                            f"({references.shape[0]:,} reference frames) on cuda:{device_index}."
                        ) from None
                    batch_frames = max(1, current_batch // 2)
                    cp.get_default_memory_pool().free_all_blocks()
                    print(
                        f"cuVS alignment OOM for speaker {speaker_id}; retrying at "
                        f"batch_frames={batch_frames:,}",
                        flush=True,
                    )
        finally:
            del index, reference_device
            cp.get_default_memory_pool().free_all_blocks()
    return indices


def resolve_alignment_backend(
    requested_backend: str,
    alignment_device: torch.device,
) -> tuple[str, tuple[Any, Any] | None]:
    """Resolve the optional GPU backend once, before the expensive alignment loop."""
    if alignment_device.type != "cuda":
        if requested_backend == "cuvs":
            raise ValueError("--alignment-backend cuvs requires a CUDA --alignment-device")
        return "sklearn", None
    if requested_backend == "torch":
        return "torch", None
    try:
        cuvs_modules = load_cuvs()
    except RuntimeError:
        if requested_backend == "cuvs":
            raise
        print("cuVS is unavailable; using the PyTorch exact cosine backend.", flush=True)
        return "torch", None
    return "cuvs", cuvs_modules


def nearest_indices(
    source: np.ndarray,
    references: np.ndarray,
    alignment_device: torch.device,
    n_jobs: int,
    alignment_batch_frames: int,
    speaker_id: str,
    alignment_backend: str,
    cuvs_modules: tuple[Any, Any] | None,
) -> np.ndarray:
    if alignment_device.type == "cuda":
        if alignment_backend == "cuvs":
            if cuvs_modules is None:
                raise AssertionError("cuVS backend resolved without its imported modules")
            return cuvs_nearest_indices(
                source,
                references,
                alignment_device,
                alignment_batch_frames,
                speaker_id,
                *cuvs_modules,
            )
        return gpu_nearest_indices(
            source, references, alignment_device, alignment_batch_frames, speaker_id
        )
    return cpu_nearest_indices(source, references, n_jobs)


def solve_utxss_blockwise(
    xs: np.ndarray,
    source_u: np.ndarray,
    singular_values: np.ndarray,
    speaker_count: int,
    feature_dim: int,
) -> np.ndarray:
    """Solve the legacy UTXSS least-squares system without flattening XS.

    The original solve is ``lstsq(XS.reshape(-1, d), repeat(U, speakers))``.
    Accumulating its normal equations a speaker block at a time has the same
    least-squares objective, while keeping a Fortran-order XS file contiguous
    during both construction and reading.  This avoids a giant C-order copy of
    XS solely for ``reshape``.
    """
    gram = np.zeros((feature_dim, feature_dim), dtype=np.float64)
    cross = np.zeros((feature_dim, source_u.shape[1]), dtype=np.float64)
    for speaker_index in range(speaker_count):
        start = speaker_index * feature_dim
        block = xs[:, start:start + feature_dim]
        # The operands are only one speaker block (about 128 MiB at the
        # default source-frame cap), never the full >300 GiB aligned matrix.
        block64 = np.asarray(block, dtype=np.float64)
        gram += block64.T @ block64
        cross += block64.T @ source_u
    utxs, _, _, _ = np.linalg.lstsq(gram, cross, rcond=None)
    return np.asarray(utxs * singular_values[np.newaxis, :], dtype=np.float32)


def clear_known_outputs(output_dir: Path) -> None:
    for name in FINAL_ARTIFACTS:
        (output_dir / name).unlink(missing_ok=True)


def existing_outputs(output_dir: Path) -> Iterable[Path]:
    return (output_dir / name for name in FINAL_ARTIFACTS if (output_dir / name).exists())


def main() -> None:
    args = parse_args()
    feature_root = args.feature_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    xs_work_dir = output_dir if args.xs_work_dir is None else args.xs_work_dir.expanduser().resolve()
    existing_xs_path = None if args.existing_xs is None else args.existing_xs.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    xs_work_dir.mkdir(parents=True, exist_ok=True)
    if args.xs_work_dir is not None and not args.discard_xs_after_factorization:
        raise ValueError(
            "--xs-work-dir requires --discard-xs-after-factorization. Publishing a huge XS "
            "across storage filesystems is intentionally unsupported."
        )
    alignment_device = torch.device(args.alignment_device)
    if alignment_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"GPU alignment requested on {alignment_device}, but CUDA is unavailable.")
        torch.cuda.set_device(alignment_device)
    alignment_backend, cuvs_modules = resolve_alignment_backend(
        args.alignment_backend, alignment_device
    )

    present = list(existing_outputs(output_dir))
    if present and not args.overwrite:
        raise FileExistsError(
            "Refusing to mix a new factorization with existing artifacts. "
            "Use a new --output-dir or pass --overwrite explicitly: "
            + ", ".join(str(path) for path in present)
        )
    if args.overwrite:
        clear_known_outputs(output_dir)

    speaker_utterances = discover_speaker_utterances(feature_root, args.speaker_id_regex)
    if args.source_speaker not in speaker_utterances:
        raise KeyError(
            f"Source speaker {args.source_speaker!r} was not found under {feature_root}; "
            f"discovered {len(speaker_utterances):,} speakers"
        )
    speakers = list(speaker_utterances.items())
    xs_partial: Path | None = None
    try:
        if existing_xs_path is not None:
            if not existing_xs_path.is_file():
                raise FileNotFoundError(f"Existing XS matrix does not exist: {existing_xs_path}")
            xs = np.load(existing_xs_path, mmap_mode="r")
            if xs.ndim != 2 or not np.issubdtype(xs.dtype, np.floating):
                raise ValueError(
                    f"Existing XS must be a rank-2 floating .npy matrix, got {xs.shape} {xs.dtype}"
                )
            source_frames = int(xs.shape[0])
            if xs.shape[1] % len(speakers) != 0:
                raise ValueError(
                    f"Existing XS width {xs.shape[1]:,} is not divisible by the {len(speakers):,} "
                    "speakers discovered from --feature-root."
                )
            feature_dim = int(xs.shape[1] // len(speakers))
            if args.max_source_frames and source_frames > args.max_source_frames:
                raise ValueError(
                    f"Existing XS has {source_frames:,} source frames, exceeding --max-source-frames="
                    f"{args.max_source_frames:,}. Use the source run's frame cap or 0."
                )
            print(
                f"Resuming from existing XS: {existing_xs_path} "
                f"(source_frames={source_frames:,}, speakers={len(speakers):,}, "
                f"feature_dim={feature_dim:,}, order={'Fortran' if xs.flags.f_contiguous else 'C'})",
                flush=True,
            )
        else:
            source = select_source_frames(
                load_speaker_matrix(args.source_speaker, speaker_utterances[args.source_speaker]),
                args.max_source_frames,
            )
            source_frames, feature_dim = source.shape
            xs_bytes = source_frames * len(speakers) * feature_dim * np.dtype(np.float32).itemsize
            xs_free_bytes = shutil.disk_usage(xs_work_dir).free
            if xs_bytes > xs_free_bytes:
                raise OSError(
                    f"Insufficient free space for transient XS in {xs_work_dir}: need "
                    f"{xs_bytes:,} bytes, have {xs_free_bytes:,} bytes."
                )
            print(
                f"Building XS: source_frames={source_frames:,}, speakers={len(speakers):,}, "
                f"feature_dim={feature_dim:,}, rank={args.rank}, alignment_backend={alignment_backend}; "
                f"transient_xs={xs_work_dir} ({xs_bytes:,} bytes required, "
                f"{xs_free_bytes:,} bytes free)",
                flush=True,
            )
            xs_partial = xs_work_dir / f".XS.{os.getpid()}.partial.npy"
            xs = np.lib.format.open_memmap(
                xs_partial,
                mode="w+",
                dtype=np.float32,
                shape=(source_frames, len(speakers) * feature_dim),
                # Writing xs[:, speaker_start:speaker_end] is contiguous in
                # Fortran order.  In C order, each of its 32,768 rows was a tiny
                # write separated by the full width of XS.
                fortran_order=True,
            )
            for speaker_index, (speaker_id, utterance_paths) in enumerate(speakers, start=1):
                references = load_speaker_matrix(speaker_id, utterance_paths)
                if references.shape[1] != feature_dim:
                    raise ValueError(
                        f"Speaker {speaker_id!r} has feature_dim={references.shape[1]}, "
                        f"expected {feature_dim}"
                    )
                indices = nearest_indices(
                    source,
                    references,
                    alignment_device,
                    args.n_jobs,
                    args.alignment_batch_frames,
                    speaker_id,
                    alignment_backend,
                    cuvs_modules,
                )
                start = (speaker_index - 1) * feature_dim
                xs[:, start:start + feature_dim] = references[indices]
                print(
                    f"Aligned speaker {speaker_index}/{len(speakers)}: {speaker_id} "
                    f"({len(utterance_paths):,} utterances; {references.shape[0]:,} reference frames)",
                    flush=True,
                )
            xs.flush()

        if args.rank > min(source_frames, len(speakers) * feature_dim):
            raise ValueError(
                f"rank={args.rank} exceeds min(source_frames={source_frames}, "
                f"speaker_count*feature_dim={len(speakers) * feature_dim})"
            )
        xs_order = "Fortran" if xs.flags.f_contiguous else "C"

        print("Computing randomized SVD...", flush=True)
        u, singular_values, vt_flat = randomized_svd(
            xs,
            n_components=args.rank,
            n_iter=args.svd_n_iter,
            random_state=args.random_seed,
        )
        u = np.asarray(u, dtype=np.float32)
        singular_values = np.asarray(singular_values, dtype=np.float32)
        vt = np.asarray(vt_flat, dtype=np.float32).reshape(args.rank, len(speakers), feature_dim).swapaxes(0, 1)
        transforms = {speaker_id: vt[index] for index, (speaker_id, _) in enumerate(speakers)}

        print("Writing SVD artifacts and speaker transforms...", flush=True)
        atomic_save_npy(u, output_dir / "U.npy")
        atomic_save_npy(singular_values, output_dir / "S.npy")
        atomic_save_npy(vt, output_dir / "VT.npy")
        atomic_save_npy(transforms, output_dir / "transforms.npy")

        print("Solving the UTXSS least-squares projection blockwise...", flush=True)
        utxss = solve_utxss_blockwise(
            xs,
            u,
            singular_values,
            len(speakers),
            feature_dim,
        )
        atomic_save_npy(utxss, output_dir / "UTXSS.npy")

        del xs
        if xs_partial is not None:
            if args.discard_xs_after_factorization:
                xs_partial.unlink()
                print(f"Discarded transient XS: {xs_partial}", flush=True)
            else:
                os.replace(xs_partial, output_dir / "XS.npy")
        atomic_save_json(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "feature_root": str(feature_root),
                "source_speaker": args.source_speaker,
                "source_utterances": len(speaker_utterances[args.source_speaker]),
                "speaker_id_regex": args.speaker_id_regex,
                "speaker_utterance_counts": {
                    speaker_id: len(utterance_paths) for speaker_id, utterance_paths in speakers
                },
                "source_frames": source_frames,
                "feature_dim": feature_dim,
                "rank": args.rank,
                "max_source_frames": args.max_source_frames,
                "svd_n_iter": args.svd_n_iter,
                "random_seed": args.random_seed,
                "alignment_device": str(alignment_device),
                "alignment_backend": alignment_backend,
                "alignment_batch_frames": args.alignment_batch_frames,
                "existing_xs": None if existing_xs_path is None else str(existing_xs_path),
                "xs_work_dir": None if xs_partial is None else str(xs_work_dir),
                "xs_persisted": bool(xs_partial is not None and not args.discard_xs_after_factorization),
                "xs_shape": [source_frames, len(speakers) * feature_dim],
                "xs_order": xs_order,
                "utxss_shape": list(utxss.shape),
                "formula": "UTXSS = lstsq(XS.reshape(-1, feature_dim), repeat(U, speakers)) @ diag(S)",
                "utxss_solver": "blockwise normal equations equivalent to the stated least-squares objective",
            },
            output_dir / "factorization-provenance.json",
        )
    except Exception:
        if xs_partial is not None:
            xs_partial.unlink(missing_ok=True)
        raise

    print(
        f"Completed factorization: {len(speakers):,} transforms and "
        f"UTXSS {tuple(utxss.shape)} written to {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
