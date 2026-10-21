#!/usr/bin/env python3
"""GPU alignment for the four-corpus, no-anchor UTXSS workflow.

The reference is exactly 32,768 WavLM-Large layer-6 frames: 8,192 frames
sampled from each of LibriSpeech train-clean-100, MSP-Podcast, Mixer6 CH01/02,
and GLOBE.  Every selected speaker receives one 32,768 x 1024 aligned block.
Those blocks are transient inputs to the following CPU factorization stage.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.signal import resample_poly
import soundfile as sf
import torch
import torch.nn.functional as F


RATE = 16_000
MAX_SECONDS = 20.0
QUOTA = 8_192
DIM = 1_024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("probe", "align"), required=True)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--batch-items", type=int, default=1)
    parser.add_argument("--max-chunk-seconds", type=float, default=MAX_SECONDS)
    parser.add_argument("--source-chunk-frames", type=int, default=4_096)
    parser.add_argument(
        "--oom-backoff-fraction",
        type=float,
        default=0.90,
        help="Retain this fraction of the current batch after an actual CUDA OOM.",
    )
    args = parser.parse_args()
    if args.batch_items < 1:
        parser.error("--batch-items must be positive")
    if args.max_chunk_seconds <= 0:
        parser.error("--max-chunk-seconds must be positive")
    if args.source_chunk_frames < 1:
        parser.error("--source-chunk-frames must be positive")
    if not 0.0 < args.oom_backoff_fraction < 1.0:
        parser.error("--oom-backoff-fraction must be in (0, 1)")
    return args


class GlobeAudio:
    """Small LRU cache of parquet row groups while reading GLOBE audio bytes."""

    def __init__(self) -> None:
        self.cache: OrderedDict[tuple[str, int], list[dict[str, object]]] = OrderedDict()

    def audio_bytes(self, record: dict[str, object]) -> bytes:
        key = (str(record["parquet_path"]), int(record["row_group"]))
        values = self.cache.get(key)
        if values is None:
            values = (
                pq.ParquetFile(key[0])
                .read_row_group(key[1], columns=["audio"])
                .column("audio")
                .to_pylist()
            )
            self.cache[key] = values
            if len(self.cache) > 8:
                self.cache.popitem(last=False)
        audio = values[int(record["row_offset"])]
        if not isinstance(audio, dict) or not isinstance(audio.get("bytes"), bytes):
            raise TypeError(f"GLOBE record has no embedded audio bytes: {record['id']}")
        return audio["bytes"]


def resample_to_16khz(audio: np.ndarray, sample_rate: int, record_id: str) -> np.ndarray:
    if sample_rate == RATE:
        return audio
    if sample_rate <= 0:
        raise ValueError(f"Invalid sample rate {sample_rate} for {record_id}")
    divisor = math.gcd(sample_rate, RATE)
    return np.asarray(
        resample_poly(audio, RATE // divisor, sample_rate // divisor).astype(np.float32, copy=False)
    )


def read_audio(record: dict[str, object], globe: GlobeAudio) -> torch.Tensor:
    start_seconds = float(record.get("start_seconds", 0.0))
    duration_seconds = float(record["duration_seconds"])
    if record["kind"] == "audio":
        with sf.SoundFile(str(record["path"])) as handle:
            start = round(start_seconds * handle.samplerate)
            frames = round(duration_seconds * handle.samplerate)
            handle.seek(start)
            audio = handle.read(frames=frames, dtype="float32", always_2d=True)
            sample_rate = handle.samplerate
    elif record["kind"] == "globe_parquet":
        audio, sample_rate = sf.read(
            io.BytesIO(globe.audio_bytes(record)), dtype="float32", always_2d=True
        )
        start = round(start_seconds * sample_rate)
        frames = round(duration_seconds * sample_rate)
        audio = audio[start : start + frames]
    else:
        raise ValueError(f"Unsupported audio kind {record['kind']!r}: {record['id']}")
    if audio.ndim != 2 or audio.shape[1] != 1:
        raise ValueError(f"Expected mono audio for {record['id']}, got {audio.shape}")
    audio = resample_to_16khz(audio[:, 0], sample_rate, str(record["id"]))
    if audio.size == 0:
        raise ValueError(f"Empty audio after reading {record['id']}")
    return torch.from_numpy(np.ascontiguousarray(audio))


def chunk_records(records: list[dict[str, object]], seconds: float) -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []
    for record in records:
        remaining = float(record["duration_seconds"])
        start = float(record.get("start_seconds", 0.0))
        chunk_index = 0
        while remaining > 1e-6:
            duration = min(remaining, seconds)
            chunk = dict(record)
            chunk["start_seconds"] = start
            chunk["duration_seconds"] = duration
            chunk["id"] = f"{record['id']}:chunk{chunk_index}"
            chunks.append(chunk)
            start += duration
            remaining -= duration
            chunk_index += 1
    return chunks


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_wavlm(device: torch.device) -> torch.nn.Module:
    model = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, device=device)
    model.eval()
    return model


def encode(
    wavlm: torch.nn.Module, waveforms: list[torch.Tensor], device: torch.device
) -> list[torch.Tensor]:
    padded_samples = max(waveform.numel() for waveform in waveforms)
    audio = torch.zeros((len(waveforms), padded_samples), device=device)
    padding_mask = torch.ones((len(waveforms), padded_samples), dtype=torch.bool, device=device)
    for index, waveform in enumerate(waveforms):
        audio[index, : waveform.numel()] = waveform.to(device, non_blocking=True)
        padding_mask[index, : waveform.numel()] = False
    with torch.inference_mode():
        features, feature_padding_mask = wavlm.extract_features(
            audio, padding_mask=padding_mask, output_layer=6
        )
    if feature_padding_mask is None:
        lengths = [features.shape[1]] * len(waveforms)
    else:
        lengths = (~feature_padding_mask).sum(dim=1).tolist()
    return [features[index, :length].float() for index, length in enumerate(lengths)]


def batches(records: list[dict[str, object]], batch_size: int) -> list[list[dict[str, object]]]:
    ordered = sorted(records, key=lambda record: float(record["duration_seconds"]))
    return [ordered[index : index + batch_size] for index in range(0, len(ordered), batch_size)]


def build_reference(
    root: Path, wavlm: torch.nn.Module, device: torch.device, batch_size: int, chunk_seconds: float
) -> torch.Tensor:
    globe = GlobeAudio()
    parts = []
    for corpus in ("librispeech", "msp", "mixer6", "globe"):
        records = chunk_records(load_jsonl(root / "references" / f"{corpus}.jsonl"), chunk_seconds)
        features: list[torch.Tensor] = []
        for batch in batches(records, batch_size):
            features.extend(encode(wavlm, [read_audio(record, globe) for record in batch], device))
            if sum(value.shape[0] for value in features) >= QUOTA * 2:
                break
        concatenated = torch.cat(features)
        if concatenated.shape[0] < QUOTA:
            raise RuntimeError(f"{corpus} reference provided only {concatenated.shape[0]} frames")
        indices = torch.linspace(0, concatenated.shape[0] - 1, QUOTA, device=device).round().long()
        parts.append(concatenated[indices])
    return torch.cat(parts).contiguous()


def atomic_save(path: Path, value: np.ndarray) -> None:
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


def next_oom_batch_size(current: int, fraction: float) -> int:
    """Back off only enough to address the observed failure, never by a fixed cap."""
    return max(1, min(current - 1, math.floor(current * fraction)))


def align_one(
    reference: torch.Tensor,
    records: list[dict[str, object]],
    wavlm: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    chunk_seconds: float,
    source_chunk_frames: int,
    oom_backoff_fraction: float,
) -> tuple[np.ndarray, int]:
    globe = GlobeAudio()
    scores = torch.full((reference.shape[0],), -torch.inf, device=device)
    aligned = torch.empty_like(reference)
    normalized_reference = F.normalize(reference, dim=1)
    seen_frames = 0
    ordered_records = sorted(
        chunk_records(records, chunk_seconds), key=lambda record: float(record["duration_seconds"])
    )
    offset = 0
    active_batch_size = batch_size
    while offset < len(ordered_records):
        batch = ordered_records[offset : offset + active_batch_size]
        try:
            raw_target = torch.cat(
                encode(wavlm, [read_audio(record, globe) for record in batch], device)
            )
        except RuntimeError as error:
            if not is_oom(error) or active_batch_size == 1:
                raise
            next_batch_size = next_oom_batch_size(active_batch_size, oom_backoff_fraction)
            print(
                f"CUDA OOM at batch_items={active_batch_size}; retrying the same records at "
                f"batch_items={next_batch_size}",
                flush=True,
            )
            active_batch_size = next_batch_size
            torch.cuda.empty_cache()
            continue
        normalized_target = F.normalize(raw_target, dim=1)
        seen_frames += raw_target.shape[0]
        for start in range(0, reference.shape[0], source_chunk_frames):
            end = min(start + source_chunk_frames, reference.shape[0])
            values, indices = (normalized_reference[start:end] @ normalized_target.T).max(dim=1)
            previous_scores = scores[start:end]
            replace = values > previous_scores
            if replace.any():
                next_scores = previous_scores.clone()
                next_aligned = aligned[start:end].clone()
                next_scores[replace] = values[replace]
                next_aligned[replace] = raw_target[indices[replace]]
                scores[start:end] = next_scores
                aligned[start:end] = next_aligned
        del raw_target, normalized_target
        offset += len(batch)
    if seen_frames == 0 or not torch.isfinite(scores).all():
        raise RuntimeError("No valid target frames were available for alignment")
    return aligned.cpu().numpy().astype(np.float32, copy=False), active_batch_size


def is_oom(error: BaseException) -> bool:
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()


def probe(wavlm: torch.nn.Module, device: torch.device, source_chunk_frames: int) -> tuple[int, int]:
    """Probe the whole inference plus all reference-search chunks at max duration."""

    reference = F.normalize(torch.randn((QUOTA * 4, DIM), device=device), dim=1)

    def succeeds(batch_size: int) -> bool:
        try:
            torch.cuda.empty_cache()
            waveforms = [torch.zeros(round(MAX_SECONDS * RATE)) for _ in range(batch_size)]
            raw_target = torch.cat(encode(wavlm, waveforms, device))
            normalized_target = F.normalize(raw_target, dim=1)
            for start in range(0, reference.shape[0], source_chunk_frames):
                end = min(start + source_chunk_frames, reference.shape[0])
                _ = (reference[start:end] @ normalized_target.T).max(dim=1)
            torch.cuda.synchronize(device)
            return True
        except RuntimeError as error:
            if not is_oom(error):
                raise
            torch.cuda.empty_cache()
            return False

    lower, upper = 0, 1
    while succeeds(upper):
        lower, upper = upper, upper * 2
    while upper - lower > 1:
        middle = (lower + upper) // 2
        if succeeds(middle):
            lower = middle
        else:
            upper = middle
    if lower == 0:
        raise RuntimeError("A single worst-case 20-second audio item does not fit")
    return max(1, math.floor(lower * 0.8)), lower


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda:0")
    wavlm = load_wavlm(device)
    if args.mode == "probe":
        if args.result is None:
            raise ValueError("--result is required for --mode probe")
        selected, maximum = probe(wavlm, device, args.source_chunk_frames)
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(
            json.dumps(
                {
                    "max_batch_items": maximum,
                    "selected_batch_items": selected,
                    "headroom_fraction": 0.20,
                    "probe_chunk_seconds": MAX_SECONDS,
                    "source_chunk_frames": args.source_chunk_frames,
                    "complete_workload_path": "WavLM extraction plus all eight source cosine-search chunks",
                },
                indent=2,
            )
            + "\n"
        )
        print(f"PROBE_COMPLETED max_batch_items={maximum} selected_batch_items={selected}", flush=True)
        return

    selection = json.loads((args.workflow_root / "selection.json").read_text())
    records_by_speaker: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in load_jsonl(args.workflow_root / "targets_audio.jsonl"):
        records_by_speaker[str(record["speaker_id"])].append(record)
    for record in load_jsonl(args.workflow_root / "targets_globe.jsonl"):
        records_by_speaker[str(record["speaker_id"])].append(record)
    reference = build_reference(
        args.workflow_root, wavlm, device, args.batch_items, args.max_chunk_seconds
    )
    atomic_save(args.workflow_root / "mixed_reference.npy", reference.cpu().numpy())
    xs_dir = args.workflow_root / "XS"
    xs_dir.mkdir(exist_ok=True)
    speakers = selection["selected_speakers"]
    active_batch_size = args.batch_items
    for index, item in enumerate(speakers, start=1):
        speaker_id = str(item["speaker_id"])
        destination = xs_dir / f"{speaker_id}.npy"
        if destination.exists():
            previous = np.load(destination, mmap_mode="r")
            if previous.shape == (QUOTA * 4, DIM) and previous.dtype == np.float32:
                print(f"[{index}/{len(speakers)}] {speaker_id}: reusing aligned block", flush=True)
                continue
            raise ValueError(f"Invalid existing aligned block: {destination} {previous.shape} {previous.dtype}")
        aligned, active_batch_size = align_one(
            reference,
            records_by_speaker[speaker_id],
            wavlm,
            device,
            active_batch_size,
            args.max_chunk_seconds,
            args.source_chunk_frames,
            args.oom_backoff_fraction,
        )
        atomic_save(destination, aligned)
        atomic_json(
            args.workflow_root / "alignment-runtime-batch.json",
            {
                "active_batch_items": active_batch_size,
                "initial_batch_items": args.batch_items,
                "oom_backoff_fraction": args.oom_backoff_fraction,
                "last_completed_speaker_index": index,
                "last_completed_speaker_id": speaker_id,
            },
        )
        print(f"[{index}/{len(speakers)}] {speaker_id}: aligned", flush=True)
    print(f"ALIGNMENT_COMPLETED speakers={len(speakers)} xs_dir={xs_dir}", flush=True)


if __name__ == "__main__":
    main(parse_args())
