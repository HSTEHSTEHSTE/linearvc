#!/usr/bin/env python3
"""Out-of-core randomized SVD and UTXSS solve over multicorpus XS blocks.

The aligned blocks remain separate while this runs.  That avoids materializing
the 800-block, roughly 100 GiB concatenated XS matrix, while producing the
same artifact layout as LinearVC's rank-75 content factorization.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


FRAMES = 32_768
DIM = 1_024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow-root", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=75)
    parser.add_argument("--oversamples", type=int, default=10)
    parser.add_argument("--n-iter", type=int, default=7)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--retain-xs", action="store_true")
    args = parser.parse_args()
    if not 0 < args.rank <= DIM:
        parser.error(f"--rank must be in [1, {DIM}]")
    if args.oversamples < 0 or args.n_iter < 0:
        parser.error("--oversamples and --n-iter must be non-negative")
    return args


def atomic_npy(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, value)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_progress(root: Path, stage: str, current: int, total: int) -> None:
    atomic_json(
        root / "factorization-progress.json",
        {
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "current": current,
            "total": total,
        },
    )


def selected_speakers(root: Path) -> list[str]:
    selection = json.loads((root / "selection.json").read_text())
    speakers = [str(item["speaker_id"]) for item in selection["selected_speakers"]]
    if len(speakers) != 800 or len(set(speakers)) != len(speakers):
        raise ValueError(f"Expected 800 unique selected speakers, found {len(speakers)}")
    return speakers


def blocks(root: Path, speakers: list[str]) -> list[Path]:
    xs_dir = root / "XS"
    result = [xs_dir / f"{speaker}.npy" for speaker in speakers]
    missing = [path for path in result if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} aligned blocks; first: {missing[0]}")
    for path in result:
        value = np.load(path, mmap_mode="r")
        if value.shape != (FRAMES, DIM) or value.dtype != np.float32:
            raise ValueError(f"Expected float32 {(FRAMES, DIM)} block, got {value.shape} {value.dtype}: {path}")
    return result


def matrix(path: Path) -> np.ndarray:
    return np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)


def qr(value: np.ndarray) -> np.ndarray:
    return np.asarray(np.linalg.qr(value, mode="reduced")[0], dtype=np.float32)


def main(args: argparse.Namespace) -> None:
    root = args.workflow_root.resolve()
    speakers = selected_speakers(root)
    paths = blocks(root, speakers)
    sketch_rank = args.rank + args.oversamples
    if sketch_rank > DIM:
        raise ValueError(f"rank + oversamples ({sketch_rank}) exceeds feature dimension {DIM}")
    rng = np.random.default_rng(args.random_seed)
    omegas = [rng.standard_normal((DIM, sketch_rank), dtype=np.float32) for _ in paths]

    print(
        f"Starting blockwise randomized SVD: blocks={len(paths)} frames={FRAMES} dim={DIM} "
        f"rank={args.rank} oversamples={args.oversamples} n_iter={args.n_iter}",
        flush=True,
    )
    sketch = np.zeros((FRAMES, sketch_rank), dtype=np.float32)
    for index, (path, omega) in enumerate(zip(paths, omegas), start=1):
        sketch += matrix(path) @ omega
        if index % 10 == 0 or index == len(paths):
            publish_progress(root, "initial_sketch", index, len(paths))
            print(f"initial_sketch {index}/{len(paths)}", flush=True)
    basis = qr(sketch)

    for iteration in range(1, args.n_iter + 1):
        power = np.zeros((FRAMES, sketch_rank), dtype=np.float32)
        for index, path in enumerate(paths, start=1):
            value = matrix(path)
            power += value @ (value.T @ basis)
            if index % 10 == 0 or index == len(paths):
                publish_progress(root, f"power_iteration_{iteration}", index, len(paths))
                print(f"power_iteration_{iteration} {index}/{len(paths)}", flush=True)
        basis = qr(power)

    covariance = np.zeros((sketch_rank, sketch_rank), dtype=np.float64)
    for index, path in enumerate(paths, start=1):
        projected = basis.T @ matrix(path)
        covariance += projected @ projected.T
        if index % 10 == 0 or index == len(paths):
            publish_progress(root, "small_covariance", index, len(paths))
            print(f"small_covariance {index}/{len(paths)}", flush=True)
    eigenvalues, left_small = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][: args.rank]
    singular_values = np.sqrt(np.clip(eigenvalues[order], 0.0, None)).astype(np.float32)
    if not np.all(singular_values > 0):
        raise RuntimeError("The randomized factorization produced a zero singular value")
    left_small = np.asarray(left_small[:, order], dtype=np.float32)
    u = np.asarray(basis @ left_small, dtype=np.float32)

    vt = np.empty((len(paths), args.rank, DIM), dtype=np.float32)
    gram = np.zeros((DIM, DIM), dtype=np.float64)
    cross = np.zeros((DIM, args.rank), dtype=np.float64)
    for index, path in enumerate(paths, start=1):
        value = matrix(path)
        projected = basis.T @ value
        vt[index - 1] = (left_small.T @ projected) / singular_values[:, np.newaxis]
        value64 = np.asarray(value, dtype=np.float64)
        gram += value64.T @ value64
        cross += value64.T @ u
        if index % 10 == 0 or index == len(paths):
            publish_progress(root, "transforms_and_utxss", index, len(paths))
            print(f"transforms_and_utxss {index}/{len(paths)}", flush=True)
    utxs, _, _, _ = np.linalg.lstsq(gram, cross, rcond=None)
    utxss = np.asarray(utxs * singular_values[np.newaxis, :], dtype=np.float32)
    transforms = {speaker: vt[index] for index, speaker in enumerate(speakers)}

    atomic_npy(root / "U.npy", u)
    atomic_npy(root / "S.npy", singular_values)
    atomic_npy(root / "VT.npy", vt)
    atomic_npy(root / "transforms.npy", transforms)
    atomic_npy(root / "UTXSS.npy", utxss)
    if u.shape != (FRAMES, args.rank) or vt.shape != (len(paths), args.rank, DIM) or utxss.shape != (DIM, args.rank):
        raise AssertionError("Published factorization artifacts have unexpected shapes")
    if not all(np.isfinite(value).all() for value in (u, singular_values, vt, utxss)):
        raise RuntimeError("Refusing to publish non-finite factorization artifacts")
    atomic_json(
        root / "factorization-provenance.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "reference": {
                "type": "no_single_anchor_mixed_reference",
                "frames_total": FRAMES,
                "frames_per_corpus": 8192,
                "corpora": ["librispeech", "msp", "mixer6", "globe"],
            },
            "speakers": len(speakers),
            "speakers_per_corpus": 200,
            "block_shape": [FRAMES, DIM],
            "rank": args.rank,
            "oversamples": args.oversamples,
            "n_iter": args.n_iter,
            "random_seed": args.random_seed,
            "svd": "blockwise randomized SVD over the conceptual horizontal concatenation of XS blocks",
            "utxss_formula": "lstsq(sum_i X_i.T X_i, sum_i X_i.T U) @ diag(S)",
            "utxss_shape": list(utxss.shape),
        },
    )
    if not args.retain_xs:
        shutil.rmtree(root / "XS")
        print(f"XS_PRUNED path={root / 'XS'}", flush=True)
    publish_progress(root, "completed", len(paths), len(paths))
    print(
        f"FACTORIZATION_COMPLETED speakers={len(paths)} utxss_shape={tuple(utxss.shape)}",
        flush=True,
    )


if __name__ == "__main__":
    main(parse_args())
