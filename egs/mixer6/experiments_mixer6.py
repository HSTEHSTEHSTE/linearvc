#!/usr/bin/env python
"""Build a rank-r content factorization from Mixer 6 manifest segments.

The source/anchor block is supplied as a precomputed WavLM feature matrix.
Each target speaker is read from a JSONL manifest, encoded in batches, and
aligned to that anchor with an exact streamed cosine nearest-neighbor search.
The streamed matcher avoids materializing a speaker's complete feature matrix.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm import tqdm


FEATURE_DIM = 1024


def check_argv():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--anchor-xs", type=Path, required=True)
    parser.add_argument("--anchor-speaker", default="103")
    parser.add_argument("--rank", type=int, default=75)
    parser.add_argument(
        "--working-rank",
        type=int,
        default=1024,
        help="Intermediate SVD rank; 1024 reproduces the LibriSpeech pipeline.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/hltcoe/xli/ARTS/linearvc/exp/content_factorization"),
    )
    parser.add_argument("--corpus-name", default="mixer6_ldc_train_local_eval")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-batch-samples",
        type=int,
        default=480_000,
        help="Upper bound for padded waveform samples in one WavLM batch.",
    )
    parser.add_argument("--max-batch-items", type=int, default=16)
    parser.add_argument("--source-chunk-frames", type=int, default=4096)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument(
        "--factorize-speakers-file",
        type=Path,
        help=(
            "Optional newline-delimited speaker order for factorization. This permits "
            "XS blocks prepared from multiple corpora to be factorized together."
        ),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write aligned XS blocks but do not run the CPU factorization.",
    )
    parser.add_argument(
        "--factorize-only",
        action="store_true",
        help="Skip WavLM extraction/alignment and factorize existing XS blocks.",
    )
    return parser.parse_args()


def load_manifest(path: Path):
    by_speaker = defaultdict(list)
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            record = json.loads(line)
            required = {"speaker_id", "path", "start_seconds", "duration_seconds"}
            missing = required.difference(record)
            if missing:
                raise ValueError(f"{path}:{line_number}: missing keys {sorted(missing)}")
            if record["duration_seconds"] <= 0:
                continue
            by_speaker[str(record["speaker_id"])].append(record)

    if not by_speaker:
        raise ValueError(f"No usable records in manifest: {path}")
    return {speaker: by_speaker[speaker] for speaker in sorted(by_speaker)}


def load_speaker_order(path: Path, anchor_speaker: str):
    speakers = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not speakers:
        raise ValueError(f"No speakers listed in {path}")
    if speakers[0] != anchor_speaker:
        raise ValueError(f"{path} must start with anchor speaker {anchor_speaker}")
    if len(set(speakers)) != len(speakers):
        raise ValueError(f"Duplicate speakers in {path}")
    return speakers


def atomic_save(path: Path, array):
    temporary_path = path.with_name(path.stem + ".tmp.npy")
    np.save(temporary_path, array)
    temporary_path.replace(path)


def load_anchor(path: Path):
    x = np.load(path, mmap_mode="r")
    if x.ndim != 2 or x.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected anchor XS shape (T, {FEATURE_DIM}), got {x.shape}: {path}")
    return np.asarray(x, dtype=np.float32)


def iter_wavlm_feature_batches(records, wavlm, device, max_batch_samples, max_batch_items):
    """Yield concatenated WavLM frames for one speaker.

    Manifest entries are length-sorted to minimize padding. Open SoundFile
    handles are reused, which avoids reopening long Mixer 6 FLAC channels for
    every speech segment.
    """

    files = {}
    prepared = []
    try:
        for record in records:
            path = record["path"]
            handle = files.get(path)
            if handle is None:
                handle = sf.SoundFile(path)
                files[path] = handle
            if handle.samplerate != 16_000 or handle.channels != 1:
                raise ValueError(
                    f"Expected mono 16 kHz audio, got {handle.channels} channels at "
                    f"{handle.samplerate} Hz: {path}"
                )
            start = int(round(record["start_seconds"] * handle.samplerate))
            frames = int(round(record["duration_seconds"] * handle.samplerate))
            if start < 0 or frames <= 0 or start + frames > handle.frames:
                raise ValueError(
                    f"Invalid segment ({start=}, {frames=}) for {path} with {handle.frames} samples"
                )
            prepared.append((frames, start, path))

        prepared.sort()
        batch = []
        batch_max_samples = 0
        for item in prepared:
            candidate_max = max(batch_max_samples, item[0])
            candidate_cost = candidate_max * (len(batch) + 1)
            if batch and (
                len(batch) >= max_batch_items or candidate_cost > max_batch_samples
            ):
                yield encode_batch(batch, files, wavlm, device)
                batch = []
                batch_max_samples = 0
            batch.append(item)
            batch_max_samples = max(batch_max_samples, item[0])
        if batch:
            yield encode_batch(batch, files, wavlm, device)
    finally:
        for handle in files.values():
            handle.close()


def encode_batch(batch, files, wavlm, device):
    max_samples = max(frames for frames, _, _ in batch)
    waveforms = torch.zeros((len(batch), max_samples), dtype=torch.float32)
    padding_mask = torch.ones((len(batch), max_samples), dtype=torch.bool)

    for index, (frames, start, path) in enumerate(batch):
        handle = files[path]
        handle.seek(start)
        audio = handle.read(frames, dtype="float32", always_2d=True)
        if audio.shape != (frames, 1):
            raise RuntimeError(f"Short read from {path}: expected {(frames, 1)}, got {audio.shape}")
        waveforms[index, :frames] = torch.from_numpy(audio[:, 0])
        padding_mask[index, :frames] = False

    with torch.inference_mode():
        features, feature_padding_mask = wavlm.extract_features(
            waveforms.to(device, non_blocking=True),
            padding_mask=padding_mask.to(device, non_blocking=True),
            output_layer=6,
        )

    if feature_padding_mask is None:
        lengths = [features.shape[1]] * len(batch)
    else:
        lengths = (~feature_padding_mask).sum(dim=1).tolist()
    return torch.cat(
        [features[index, :length] for index, length in enumerate(lengths)], dim=0
    )


def align_streaming(source_features, feature_batches, device, source_chunk_frames):
    """Return the exact cosine-NN alignment without retaining all references."""

    source = torch.from_numpy(source_features).to(device)
    source_normalized = F.normalize(source, dim=1)
    num_source_frames = source.shape[0]
    best_scores = torch.full((num_source_frames,), -torch.inf, device=device)
    aligned = torch.empty_like(source)
    seen_reference_frames = 0

    for references in feature_batches:
        if references.numel() == 0:
            continue
        references_normalized = F.normalize(references, dim=1)
        seen_reference_frames += references.shape[0]
        for start in range(0, num_source_frames, source_chunk_frames):
            end = min(start + source_chunk_frames, num_source_frames)
            scores = source_normalized[start:end] @ references_normalized.T
            values, indices = scores.max(dim=1)
            replace = values > best_scores[start:end]
            if replace.any():
                source_indices = torch.arange(start, end, device=device)[replace]
                best_scores[source_indices] = values[replace]
                aligned[source_indices] = references[indices[replace]]

    if seen_reference_frames == 0 or not torch.isfinite(best_scores).all():
        raise RuntimeError("No valid target feature frames were available for alignment")
    return aligned.cpu().numpy().astype(np.float32, copy=False), seen_reference_frames


def svd_low_rank_update(U, S, VT, P, Q, eps=1e-12):
    """Brand-style low-rank SVD update, kept compatible with experiments_libri.py."""

    m, k = U.shape
    k2, n = VT.shape
    if k != k2 or P.shape[0] != m or Q.shape[0] != n:
        raise ValueError("Incompatible low-rank SVD update shapes")
    r = P.shape[1]
    if Q.shape[1] != r:
        raise ValueError("P and Q must have the same update rank")

    P = P.astype(np.float64, copy=False)
    Q = Q.astype(np.float64, copy=False)
    P_proj = U.T @ P
    Q_proj = VT @ Q
    P_perp = P - U @ P_proj
    Q_perp = Q - VT.T @ Q_proj

    if np.linalg.norm(P_perp) > eps:
        Up, Rp = np.linalg.qr(P_perp, mode="reduced")
    else:
        Up = np.zeros((m, 0), dtype=np.float64)
        Rp = np.zeros((0, r), dtype=np.float64)
    if np.linalg.norm(Q_perp) > eps:
        Vp, Rq = np.linalg.qr(Q_perp, mode="reduced")
    else:
        Vp = np.zeros((n, 0), dtype=np.float64)
        Rq = np.zeros((0, r), dtype=np.float64)

    core = np.zeros((k + Up.shape[1], k + Vp.shape[1]), dtype=np.float64)
    core[:k, :k] = np.diag(S)
    core += np.vstack([P_proj, Rp]) @ np.vstack([Q_proj, Rq]).T
    U_core, S_new, VT_core = np.linalg.svd(core, full_matrices=False)
    U_new = np.column_stack([U, Up]) @ U_core
    VT_new = (np.column_stack([VT.T, Vp]) @ VT_core.T).T
    return U_new, S_new, VT_new


def factorize(xs_dir: Path, speakers, rank, working_rank, output_dir: Path):
    blocks = []
    expected_shape = None
    for speaker in speakers:
        path = xs_dir / f"{speaker}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Missing aligned block for speaker {speaker}: {path}")
        x = np.load(path, mmap_mode="r")
        if x.ndim != 2 or x.shape[1] != FEATURE_DIM:
            raise ValueError(f"Unexpected block shape {x.shape}: {path}")
        if expected_shape is None:
            expected_shape = x.shape
        elif x.shape != expected_shape:
            raise ValueError(f"All blocks must share a shape; expected {expected_shape}, got {x.shape}")
        blocks.append(path)

    block_frames, block_dim = expected_shape
    print(f"Factorizing {len(blocks)} blocks with shape {expected_shape}; working rank={working_rank}")
    initial = np.asarray(np.load(blocks[0]), dtype=np.float64)
    U, S, VT = np.linalg.svd(initial, full_matrices=False)
    keep = min(working_rank, len(S))
    U, S, VT = U[:, :keep], S[:keep], VT[:keep, :]

    for index, path in enumerate(blocks[1:], start=1):
        x = np.asarray(np.load(path), dtype=np.float64)
        VT = np.hstack([VT, np.zeros((VT.shape[0], block_dim), dtype=np.float64)])
        q = np.zeros((VT.shape[1], block_dim), dtype=np.float64)
        q[-block_dim:, :] = np.eye(block_dim, dtype=np.float64)
        started = time.time()
        U, S, VT = svd_low_rank_update(U, S, VT, x, q)
        keep = min(working_rank, len(S))
        U, S, VT = U[:, :keep], S[:keep], VT[:keep, :]
        print(f"SVD update {index}/{len(blocks) - 1}: {path.stem} ({time.time() - started:.1f}s)")

    output_rank = min(rank, len(S))
    U = U[:, :output_rank].astype(np.float32)
    S = S[:output_rank].astype(np.float32)
    VT = VT[:output_rank, :].astype(np.float32)
    expected_width = len(speakers) * block_dim
    if VT.shape[1] != expected_width:
        raise ValueError(f"Expected VT width {expected_width}, got {VT.shape[1]}")
    VT_by_speaker = VT.reshape(output_rank, len(speakers), block_dim).swapaxes(0, 1)
    transforms = {speaker: VT_by_speaker[i] for i, speaker in enumerate(speakers)}

    atomic_save(output_dir / "U.npy", U)
    atomic_save(output_dir / "S.npy", S)
    atomic_save(output_dir / "VT.npy", VT_by_speaker)
    atomic_save(output_dir / "transforms.npy", transforms)
    atomic_save(output_dir / "used_speakers.npy", np.array(speakers, dtype=object))
    print(f"Wrote factorization to: {output_dir}")


def main(args):
    if args.prepare_only and args.factorize_only:
        raise ValueError("--prepare-only and --factorize-only cannot be used together")
    if not args.factorize_only and args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.rank <= 0 or args.working_rank < args.rank:
        raise ValueError("Require positive rank and --working-rank >= --rank")

    manifest_speakers = load_manifest(args.manifest)
    if args.anchor_speaker in manifest_speakers:
        raise ValueError(f"Anchor speaker {args.anchor_speaker} is unexpectedly present in the Mixer manifest")
    speakers = [args.anchor_speaker, *manifest_speakers]
    factor_speakers = (
        load_speaker_order(args.factorize_speakers_file, args.anchor_speaker)
        if args.factorize_speakers_file is not None
        else speakers
    )
    output_dir = (
        args.output_root
        / args.corpus_name
        / f"rank_{args.rank}"
        / f"src_{args.anchor_speaker}"
    )
    xs_dir = output_dir / "XS"
    xs_dir.mkdir(parents=True, exist_ok=True)
    print(f"Manifest speakers: {len(manifest_speakers)}; total blocks including anchor: {len(speakers)}")
    print(f"Output: {output_dir}")

    if not args.factorize_only:
        anchor = load_anchor(args.anchor_xs)
        anchor_path = xs_dir / f"{args.anchor_speaker}.npy"
        if anchor_path.is_file():
            existing = np.load(anchor_path, mmap_mode="r")
            if existing.shape != anchor.shape:
                raise ValueError(f"Existing anchor block has shape {existing.shape}, expected {anchor.shape}")
        else:
            atomic_save(anchor_path, anchor)
            print(f"Copied anchor block: {anchor_path}")

        wavlm = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, device=args.device)
        wavlm.eval()
        for index, speaker in enumerate(manifest_speakers, start=1):
            output_path = xs_dir / f"{speaker}.npy"
            if output_path.is_file():
                x = np.load(output_path, mmap_mode="r")
                if x.shape == anchor.shape:
                    print(f"[{index}/{len(manifest_speakers)}] {speaker}: reusing {output_path}")
                    continue
                raise ValueError(f"Existing block has shape {x.shape}, expected {anchor.shape}: {output_path}")

            started = time.time()
            batches = iter_wavlm_feature_batches(
                manifest_speakers[speaker],
                wavlm,
                args.device,
                args.max_batch_samples,
                args.max_batch_items,
            )
            aligned, target_frames = align_streaming(
                anchor, batches, args.device, args.source_chunk_frames
            )
            atomic_save(output_path, aligned)
            if args.print_every > 0 and index % args.print_every == 0:
                print(
                    f"[{index}/{len(manifest_speakers)}] {speaker}: "
                    f"target_frames={target_frames} elapsed={time.time() - started:.1f}s"
                )

    if not args.prepare_only:
        factorize(xs_dir, factor_speakers, args.rank, args.working_rank, output_dir)


if __name__ == "__main__":
    main(check_argv())
