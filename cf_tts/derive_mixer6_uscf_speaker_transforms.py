#!/usr/bin/env python3
"""Probe and derive offline-WavLM USCF transforms for retained Mixer6 speakers.

This program uses only Microsoft's original WavLM-Large implementation and
official checkpoint.  It never writes extracted WavLM tensors: bounded audio
batches are transformed and immediately accumulated into the least-squares
sufficient statistics for each speaker matrix.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


CONVERTER_VERSION = 2
SAMPLE_RATE = 16_000
WAVLM_DIM = 1024
CONTENT_DIM = 75
OFFICIAL_CHECKPOINT_SHA256 = "6fb4b3c3e6aa567f0a997b30855859cb81528ee8078802af439f7b2da0bf100f"
SPEAKER_RE = re.compile(r"_(?:LDC|HRM)_(\d+)_CH\d+\.flac$", re.IGNORECASE)
PROVENANCE_NAME = "speaker-transform-provenance.json"
STATS_NAME = "speaker-transform-stats.json"


@dataclass(frozen=True)
class AudioFile:
    path: Path
    size_bytes: int
    source_frames: int
    sample_rate: int


@dataclass(frozen=True)
class ChunkSpec:
    audio: AudioFile
    start_frame: int
    source_frames: int
    resampled_frames: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("probe", "derive", "pack"), required=True)
    parser.add_argument("--mixer6-root", type=Path, required=True)
    parser.add_argument("--projection-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--official-wavlm-source", type=Path, required=True)
    parser.add_argument("--wavlm-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--audio-chunk-seconds", type=float, default=20.0)
    parser.add_argument("--length-bucket-seconds", type=float, default=1.0)
    parser.add_argument("--progress-seconds", type=float, default=30.0)
    parser.add_argument("--probe-result", type=Path)
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--safety-fraction", type=float, default=0.85)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite-speakers", action="store_true")
    args = parser.parse_args()
    if args.audio_chunk_seconds <= 0 or args.length_bucket_seconds <= 0 or args.progress_seconds <= 0:
        parser.error("chunk, bucket, and progress durations must be positive")
    if args.max_batch_size < 1:
        parser.error("--max-batch-size must be at least one")
    if not 0 < args.safety_fraction < 1:
        parser.error("--safety-fraction must be strictly between zero and one")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    if args.mode == "probe" and args.probe_result is None:
        parser.error("--mode probe requires --probe-result")
    if args.mode == "derive" and args.probe_result is None:
        parser.error("--mode derive requires the successful --probe-result")
    return args


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


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
    days, seconds = divmod(seconds, 86_400)
    hours, seconds = divmod(seconds, 3_600)
    minutes, seconds = divmod(seconds, 60)
    return f"{days}d {hours:02d}:{minutes:02d}:{seconds:02d}" if days else f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def load_official_wavlm(source_root: Path, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Use the sanctioned UniLM loader, never a Hugging Face model."""
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from official_wavlm import load_official_wavlm_large  # pylint: disable=import-outside-toplevel

    actual_sha256 = sha256_file(checkpoint_path)
    if actual_sha256 != OFFICIAL_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"Official WavLM checkpoint digest mismatch: expected {OFFICIAL_CHECKPOINT_SHA256}, got {actual_sha256}"
        )
    model, _ = load_official_wavlm_large(checkpoint_path, device=device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.eval()


def inventory_mixer6(mixer6_root: Path) -> tuple[list[str], dict[str, list[AudioFile]], int]:
    roster_path = mixer6_root / "docs" / "mx6_subjs.csv"
    audio_root = mixer6_root / "data" / "pcm_flac"
    if not roster_path.is_file() or not audio_root.is_dir():
        raise FileNotFoundError(f"Expected {roster_path} and {audio_root}")
    with roster_path.open(newline="") as handle:
        roster = sorted({row["subjid"].strip() for row in csv.DictReader(handle) if row.get("subjid", "").strip()})
    enrolled = set(roster)
    grouped: dict[str, list[AudioFile]] = defaultdict(list)
    skipped = 0
    for path in sorted(audio_root.rglob("*.flac")):
        match = SPEAKER_RE.search(path.name)
        if not match or match.group(1) not in enrolled:
            skipped += 1
            continue
        info = sf.info(path)
        if info.frames <= 0 or info.samplerate <= 0:
            raise RuntimeError(f"Invalid audio metadata for {path}")
        grouped[match.group(1)].append(
            AudioFile(path=path, size_bytes=path.stat().st_size, source_frames=info.frames, sample_rate=info.samplerate)
        )
    available = [speaker for speaker in roster if grouped.get(speaker)]
    absent = len(roster) - len(available)
    if not available:
        raise RuntimeError("No enrolled Mixer6 speaker FLACs were found")
    print(
        f"[inventory] speakers={len(available):,}/{len(roster):,}; absent={absent:,} (ARTS-eval excluded); "
        f"files={sum(map(len, grouped.values())):,}; skipped={skipped:,}",
        flush=True,
    )
    return available, grouped, absent


def make_buckets(audio_files: Iterable[AudioFile], chunk_seconds: float, bucket_seconds: float) -> dict[int, list[ChunkSpec]]:
    buckets: dict[int, list[ChunkSpec]] = defaultdict(list)
    bucket_width = max(1, round(bucket_seconds * SAMPLE_RATE))
    for audio in audio_files:
        source_chunk_frames = max(1, round(chunk_seconds * audio.sample_rate))
        for start in range(0, audio.source_frames, source_chunk_frames):
            source_frames = min(source_chunk_frames, audio.source_frames - start)
            resampled_frames = round(source_frames * SAMPLE_RATE / audio.sample_rate)
            # Bucket from precomputed SoundFile frame metadata. Every full
            # chunk shares a bucket; trailing chunks are grouped by one-second
            # lengths to limit padding without materializing features.
            bucket = (resampled_frames + bucket_width - 1) // bucket_width
            buckets[bucket].append(ChunkSpec(audio, start, source_frames, resampled_frames))
    return buckets


def read_resampled_chunk(handle: sf.SoundFile, spec: ChunkSpec) -> torch.Tensor:
    handle.seek(spec.start_frame)
    array = handle.read(frames=spec.source_frames, dtype="float32", always_2d=True)
    if not len(array):
        return torch.empty(0, dtype=torch.float32)
    waveform = torch.from_numpy(np.asarray(array).mean(axis=1, dtype=np.float32))
    if handle.samplerate != SAMPLE_RATE:
        waveform = F.interpolate(
            waveform.view(1, 1, -1),
            size=spec.resampled_frames,
            mode="linear",
            align_corners=False,
        ).view(-1)
    return waveform.contiguous()


def accumulate_waveform_batch(
    wavlm: torch.nn.Module,
    waveforms: list[torch.Tensor],
    projection: torch.Tensor,
    gram: torch.Tensor,
    cross: torch.Tensor,
    device: torch.device,
) -> int:
    waveforms = [waveform for waveform in waveforms if waveform.numel() >= 400]
    if not waveforms:
        return 0
    lengths = torch.tensor([waveform.numel() for waveform in waveforms], dtype=torch.long, device=device)
    padded = pad_sequence(waveforms, batch_first=True).to(device, non_blocking=False)
    padding_mask = torch.arange(padded.shape[1], device=device).unsqueeze(0) >= lengths.unsqueeze(1)
    with torch.inference_mode():
        features, feature_padding_mask = wavlm.extract_features(padded, padding_mask=padding_mask, output_layer=6)
        if features.ndim != 3 or features.shape[0] != len(waveforms) or features.shape[2] != WAVLM_DIM:
            raise RuntimeError(f"Unexpected WavLM layer-6 shape: {tuple(features.shape)}")
        total_frames = 0
        for index in range(features.shape[0]):
            source = features[index]
            if feature_padding_mask is not None:
                source = source[~feature_padding_mask[index]]
            if not source.numel():
                continue
            content = source @ projection
            source64 = source.to(torch.float64)
            content64 = content.to(torch.float64)
            gram.add_(content64.T @ content64)
            cross.add_(content64.T @ source64)
            total_frames += int(source.shape[0])
    return total_frames


def select_probe_waveform(grouped: dict[str, list[AudioFile]], chunk_seconds: float) -> torch.Tensor:
    target_samples = round(chunk_seconds * SAMPLE_RATE)
    for audio in sorted((item for items in grouped.values() for item in items), key=lambda item: item.source_frames, reverse=True):
        source_frames = min(audio.source_frames, round(chunk_seconds * audio.sample_rate))
        spec = ChunkSpec(audio, 0, source_frames, round(source_frames * SAMPLE_RATE / audio.sample_rate))
        if spec.resampled_frames < target_samples:
            continue
        with sf.SoundFile(audio.path) as handle:
            waveform = read_resampled_chunk(handle, spec)
        if waveform.numel() >= target_samples:
            return waveform[:target_samples]
    raise RuntimeError(f"Could not find a Mixer6 recording with a {chunk_seconds:g}-second chunk for the batch probe")


def probe_candidate(
    wavlm: torch.nn.Module, waveform: torch.Tensor, projection: torch.Tensor, batch_size: int, device: torch.device,
) -> tuple[bool, int]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    gram = torch.zeros((CONTENT_DIM, CONTENT_DIM), dtype=torch.float64, device=device)
    cross = torch.zeros((CONTENT_DIM, WAVLM_DIM), dtype=torch.float64, device=device)
    try:
        accumulate_waveform_batch(wavlm, [waveform] * batch_size, projection, gram, cross, device)
        torch.cuda.synchronize(device)
        return True, int(torch.cuda.max_memory_allocated(device))
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return False, int(torch.cuda.max_memory_allocated(device))
    finally:
        del gram, cross
        torch.cuda.empty_cache()


def run_probe(args: argparse.Namespace, wavlm: torch.nn.Module, projection: torch.Tensor, grouped: dict[str, list[AudioFile]], device: torch.device) -> None:
    waveform = select_probe_waveform(grouped, args.audio_chunk_seconds)
    print(
        f"[probe] full workload path: {args.audio_chunk_seconds:g}s / {waveform.numel():,} samples / "
        f"{1 + (waveform.numel() - 400) // 320:,} frontend frames.",
        flush=True,
    )
    results: dict[int, dict[str, Any]] = {}

    def fits(candidate: int) -> bool:
        ok, peak = probe_candidate(wavlm, waveform, projection, candidate, device)
        results[candidate] = {"fits": ok, "peak_allocated_bytes": peak}
        print(f"[probe] batch={candidate}: {'fits' if ok else 'OOM'}; peak={format_bytes(peak)}", flush=True)
        return ok

    if not fits(1):
        raise RuntimeError("The representative batch size of one does not fit on the allocated GPU")
    lower, upper = 1, None
    candidate = 2
    while candidate <= args.max_batch_size:
        if fits(candidate):
            lower = candidate
            candidate *= 2
        else:
            upper = candidate
            break
    if upper is None:
        maximum = lower
    else:
        low, high = lower + 1, upper - 1
        maximum = lower
        while low <= high:
            middle = (low + high) // 2
            if fits(middle):
                maximum = middle
                low = middle + 1
            else:
                high = middle - 1
    selected = max(1, math.floor(maximum * args.safety_fraction))
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "converter": "linearvc.cf_tts.derive_mixer6_uscf_speaker_transforms",
        "converter_version": CONVERTER_VERSION,
        "official_wavlm_checkpoint": str(args.wavlm_checkpoint.resolve()),
        "official_wavlm_checkpoint_sha256": OFFICIAL_CHECKPOINT_SHA256,
        "projection_path": str(args.projection_path.resolve()),
        "projection_sha256": sha256_array(projection.detach().cpu().numpy()),
        "audio_chunk_seconds": args.audio_chunk_seconds,
        "length_bucket_seconds": args.length_bucket_seconds,
        "probed_max_batch_size": maximum,
        "safety_fraction": args.safety_fraction,
        "selected_batch_size": selected,
        "candidates": results,
        "persistent_feature_files": False,
    }
    save_json_atomically(result, args.probe_result.resolve())
    print(f"[probe] selected safe batch size: {selected}; wrote {args.probe_result.resolve()}", flush=True)


def load_probe_result(args: argparse.Namespace, projection: torch.Tensor) -> tuple[int, dict[str, Any]]:
    path = args.probe_result.resolve()
    if not path.is_file():
        raise RuntimeError(f"Missing required successful batch-probe metadata: {path}")
    result = json.loads(path.read_text())
    expected = {
        "converter_version": CONVERTER_VERSION,
        "official_wavlm_checkpoint": str(args.wavlm_checkpoint.resolve()),
        "official_wavlm_checkpoint_sha256": OFFICIAL_CHECKPOINT_SHA256,
        "projection_path": str(args.projection_path.resolve()),
        "projection_sha256": sha256_array(projection.detach().cpu().numpy()),
        "audio_chunk_seconds": args.audio_chunk_seconds,
        "length_bucket_seconds": args.length_bucket_seconds,
    }
    mismatches = {key: (result.get(key), value) for key, value in expected.items() if result.get(key) != value}
    if mismatches:
        raise RuntimeError(f"Batch-probe metadata is not compatible with this derivation: {mismatches}")
    batch_size = result.get("selected_batch_size")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise RuntimeError(f"Invalid selected_batch_size in {path}: {batch_size!r}")
    return batch_size, result


def derive_transforms(
    args: argparse.Namespace, wavlm: torch.nn.Module, projection_np: np.ndarray, projection: torch.Tensor,
    speakers: list[str], grouped: dict[str, list[AudioFile]], device: torch.device,
) -> None:
    batch_size, probe_result = load_probe_result(args, projection)
    output_dir = args.output_dir.resolve()
    provenance = {
        "converter": "linearvc.cf_tts.derive_mixer6_uscf_speaker_transforms",
        "converter_version": CONVERTER_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mixer6_root": str(args.mixer6_root.resolve()),
        "projection_path": str(args.projection_path.resolve()),
        "projection_shape": list(projection_np.shape),
        "projection_dtype": str(projection_np.dtype),
        "projection_sha256": sha256_array(projection_np),
        "official_wavlm_source": str(args.official_wavlm_source.resolve()),
        "official_wavlm_checkpoint": str(args.wavlm_checkpoint.resolve()),
        "official_wavlm_checkpoint_sha256": OFFICIAL_CHECKPOINT_SHA256,
        "wavlm_layer": 6,
        "sample_rate": SAMPLE_RATE,
        "audio_chunk_seconds": args.audio_chunk_seconds,
        "length_bucket_seconds": args.length_bucket_seconds,
        "batch_probe_result": str(args.probe_result.resolve()),
        "selected_batch_size": batch_size,
        "all_matching_mixer6_channels": True,
        "speaker_transform_formula": "pinv((X @ U).T @ (X @ U)) @ ((X @ U).T @ X)",
        "speaker_transform_dtype": "float32",
        "persistent_feature_files": False,
    }
    provenance_path = output_dir / PROVENANCE_NAME
    if provenance_path.exists():
        existing = json.loads(provenance_path.read_text())
        keys = tuple(key for key in provenance if key != "created_at_utc")
        differences = {key: (existing.get(key), provenance[key]) for key in keys if existing.get(key) != provenance[key]}
        if differences:
            raise RuntimeError(f"Existing output provenance differs from this request: {differences}")
    else:
        save_json_atomically(provenance, provenance_path)

    shard_speakers = speakers[args.shard_index::args.num_shards]
    speaker_paths = {speaker: output_dir / "speakers" / f"{speaker}.npy" for speaker in shard_speakers}
    pending = [speaker for speaker in shard_speakers if args.overwrite_speakers or not speaker_paths[speaker].is_file()]
    total_bytes = sum(audio.size_bytes for speaker in pending for audio in grouped[speaker])
    print(
        f"[derive] shard={args.shard_index + 1}/{args.num_shards}; speakers={len(shard_speakers):,}; "
        f"pending={len(pending):,}; batch={batch_size}; "
        f"input={format_bytes(total_bytes)}; probe max={probe_result['probed_max_batch_size']}",
        flush=True,
    )
    shard_dir = output_dir / "shards"
    stats_path = shard_dir / f"speaker-transform-stats-{args.shard_index + 1:02d}-of-{args.num_shards:02d}.json"
    stats: dict[str, dict[str, Any]] = json.loads(stats_path.read_text()) if stats_path.is_file() and not args.overwrite_speakers else {}
    legacy_stats_path = output_dir / STATS_NAME
    legacy_stats: dict[str, dict[str, Any]] = json.loads(legacy_stats_path.read_text()) if legacy_stats_path.is_file() and not args.overwrite_speakers else {}
    for speaker in shard_speakers:
        if not args.overwrite_speakers and speaker_paths[speaker].is_file() and speaker not in stats:
            # The pre-sharding run wrote compatible per-speaker matrices and a
            # shared stats file. Preserve those entries so sharded resumption
            # does not recompute completed speakers merely to recreate stats.
            stats[speaker] = legacy_stats.get(
                speaker,
                {"transform_path": str(speaker_paths[speaker]), "resumed_without_prior_stats": True},
            )
    started = last_report = time.monotonic()
    completed_bytes = completed_chunks = completed_frames = completed_speakers = 0

    for speaker in pending:
        audio_files = grouped[speaker]
        buckets = make_buckets(audio_files, args.audio_chunk_seconds, args.length_bucket_seconds)
        gram = torch.zeros((CONTENT_DIM, CONTENT_DIM), dtype=torch.float64, device=device)
        cross = torch.zeros((CONTENT_DIM, WAVLM_DIM), dtype=torch.float64, device=device)
        speaker_frames = speaker_chunks = 0
        processed_source_frames: dict[Path, int] = defaultdict(int)
        print(f"[speaker] {speaker}: {len(audio_files):,} FLACs; {sum(map(len, buckets.values())):,} chunks; batch={batch_size}.", flush=True)
        with ExitStack() as stack:
            handles = {audio.path: stack.enter_context(sf.SoundFile(audio.path)) for audio in audio_files}
            for bucket in sorted(buckets, reverse=True):
                specs = buckets[bucket]
                for begin in range(0, len(specs), batch_size):
                    batch_specs = specs[begin:begin + batch_size]
                    waveforms = [read_resampled_chunk(handles[spec.audio.path], spec) for spec in batch_specs]
                    speaker_frames += accumulate_waveform_batch(wavlm, waveforms, projection, gram, cross, device)
                    speaker_chunks += len(batch_specs)
                    completed_chunks += len(batch_specs)
                    for spec in batch_specs:
                        processed_source_frames[spec.audio.path] += spec.source_frames
                        completed_bytes += spec.audio.size_bytes * spec.source_frames / spec.audio.source_frames
                    if time.monotonic() - last_report >= args.progress_seconds:
                        elapsed = max(time.monotonic() - started, 1e-9)
                        rate = completed_bytes / elapsed
                        eta = (total_bytes - completed_bytes) / rate if rate else math.inf
                        percent = 100.0 * completed_bytes / total_bytes if total_bytes else 100.0
                        print(
                        f"[progress] shard={args.shard_index + 1}/{args.num_shards} speakers={completed_speakers}/{len(pending)} chunks={completed_chunks:,} "
                            f"input={format_bytes(completed_bytes)}/{format_bytes(total_bytes)} ({percent:.1f}%) "
                            f"frames={completed_frames + speaker_frames:,} throughput={format_bytes(rate)}/s ETA={format_duration(eta)}",
                            flush=True,
                        )
                        last_report = time.monotonic()
        if any(processed_source_frames[audio.path] != audio.source_frames for audio in audio_files):
            raise RuntimeError(f"Chunk coverage mismatch for Mixer6 speaker {speaker}")
        if not speaker_frames:
            raise RuntimeError(f"No valid WavLM frames were extracted for Mixer6 speaker {speaker}")
        transform = torch.linalg.pinv(gram) @ cross
        save_numpy_atomically(transform.to(torch.float32).cpu().numpy(), speaker_paths[speaker])
        stats[speaker] = {
            "files": len(audio_files),
            "chunks": speaker_chunks,
            "frames": speaker_frames,
            "batch_size": batch_size,
            "gram_rank": int(torch.linalg.matrix_rank(gram).item()),
            "gram_condition_number": float(torch.linalg.cond(gram).item()),
            "transform_path": str(speaker_paths[speaker]),
        }
        save_json_atomically(stats, stats_path)
        completed_speakers += 1
        completed_frames += speaker_frames
        print(f"[speaker] {speaker}: saved {speaker_paths[speaker]} ({speaker_frames:,} frames).", flush=True)

    missing = [speaker for speaker in shard_speakers if not speaker_paths[speaker].is_file()]
    if missing:
        raise RuntimeError(f"Shard has {len(missing)} missing matrices, e.g. {', '.join(missing[:10])}")
    completion = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "speakers": shard_speakers,
        "speaker_count": len(shard_speakers),
        "stats_path": str(stats_path),
        "selected_batch_size": batch_size,
    }
    completion_path = shard_dir / f"shard-{args.shard_index + 1:02d}-of-{args.num_shards:02d}.complete.json"
    save_json_atomically(completion, completion_path)
    print(f"Completed shard {args.shard_index + 1}/{args.num_shards}: {len(shard_speakers):,} transforms; marker: {completion_path}", flush=True)


def pack_transforms(args: argparse.Namespace) -> None:
    """Combine only verified, complete shard outputs; this is CPU-only."""
    output_dir = args.output_dir.resolve()
    shard_dir = output_dir / "shards"
    speakers: list[str] = []
    stats: dict[str, Any] = {}
    for shard_index in range(args.num_shards):
        completion_path = shard_dir / f"shard-{shard_index + 1:02d}-of-{args.num_shards:02d}.complete.json"
        if not completion_path.is_file():
            raise RuntimeError(f"Missing successful shard marker: {completion_path}")
        completion = json.loads(completion_path.read_text())
        if completion.get("num_shards") != args.num_shards or completion.get("shard_index") != shard_index:
            raise RuntimeError(f"Shard marker does not match expected shard identity: {completion_path}")
        shard_speakers = completion.get("speakers")
        if not isinstance(shard_speakers, list) or not all(isinstance(speaker, str) for speaker in shard_speakers):
            raise RuntimeError(f"Invalid speaker list in {completion_path}")
        speakers.extend(shard_speakers)
        stats_path = Path(completion.get("stats_path", ""))
        if not stats_path.is_file():
            raise RuntimeError(f"Missing shard statistics: {stats_path}")
        shard_stats = json.loads(stats_path.read_text())
        stats.update(shard_stats)
    if len(speakers) != len(set(speakers)):
        raise RuntimeError("Shard speaker lists overlap; refusing to create transforms.npy")
    transforms: dict[str, np.ndarray] = {}
    for speaker in sorted(speakers):
        path = output_dir / "speakers" / f"{speaker}.npy"
        if not path.is_file():
            raise RuntimeError(f"Missing speaker matrix named by complete shard marker: {path}")
        matrix = np.load(path)
        if matrix.shape != (CONTENT_DIM, WAVLM_DIM):
            raise RuntimeError(f"Invalid matrix for {speaker}: {path} has shape {matrix.shape}")
        transforms[speaker] = matrix
    if set(stats) != set(transforms):
        raise RuntimeError("Shard statistics do not exactly cover the packed speaker transforms")
    save_numpy_atomically(transforms, output_dir / "transforms.npy")
    save_json_atomically(stats, output_dir / STATS_NAME)
    print(f"Packed {len(transforms):,} verified Mixer6 transforms: {output_dir / 'transforms.npy'}", flush=True)


def main() -> None:
    args = parse_args()
    args.mixer6_root = args.mixer6_root.expanduser().resolve()
    args.projection_path = args.projection_path.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.official_wavlm_source = args.official_wavlm_source.expanduser().resolve()
    args.wavlm_checkpoint = args.wavlm_checkpoint.expanduser().resolve()
    if args.mode == "pack":
        pack_transforms(args)
        return
    if not args.projection_path.is_file() or not args.official_wavlm_source.is_dir() or not args.wavlm_checkpoint.is_file():
        raise SystemExit("Missing UTXSS projection, official UniLM WavLM source, or official WavLM checkpoint")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("This workflow requires an allocated CUDA GPU")
    torch.cuda.set_device(device)
    projection_np = np.asarray(np.load(args.projection_path), dtype=np.float32)
    if projection_np.shape != (WAVLM_DIM, CONTENT_DIM):
        raise SystemExit(f"Expected UTXSS shape (1024, 75), got {projection_np.shape}")
    projection = torch.from_numpy(projection_np).to(device)
    speakers, grouped, _ = inventory_mixer6(args.mixer6_root)
    print(f"[model] loading sanctioned original WavLM-Large from {args.wavlm_checkpoint}", flush=True)
    wavlm = load_official_wavlm(args.official_wavlm_source, args.wavlm_checkpoint, device)
    print(f"[model] GPU={torch.cuda.get_device_name(device)}; layer=6; UTXSS={sha256_array(projection_np)}", flush=True)
    if args.mode == "probe":
        run_probe(args, wavlm, projection, grouped, device)
    else:
        derive_transforms(args, wavlm, projection_np, projection, speakers, grouped, device)


if __name__ == "__main__":
    main()
