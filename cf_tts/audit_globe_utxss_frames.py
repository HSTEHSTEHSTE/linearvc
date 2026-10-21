#!/usr/bin/env python3
"""Audit GLOBE training speakers for rank-r UTXSS source-frame eligibility.

Only the Parquet ``speaker_id`` and official ``duration`` columns are read;
embedded audio is never decoded or materialized.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from pathlib import Path

import pyarrow.parquet as pq

SAMPLE_RATE = 16_000
RECEPTIVE_FIELD = 400
HOP = 320


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--min-frames", type=int, default=32_768)
    parser.add_argument("--rank", type=int, default=75)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=65_536)
    return parser.parse_args()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def count_frames(seconds: float) -> tuple[int, int]:
    samples = round(seconds * SAMPLE_RATE)
    frames = 0 if samples < RECEPTIVE_FIELD else 1 + (samples - RECEPTIVE_FIELD) // HOP
    return samples, frames


def scan_shard(path: Path, batch_size: int) -> tuple[dict[str, list[int]], int, int]:
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    rows = invalid = 0
    parquet = pq.ParquetFile(path)
    if not {"speaker_id", "duration"}.issubset(parquet.schema_arrow.names):
        raise ValueError(f"{path} lacks speaker_id/duration")
    for batch in parquet.iter_batches(columns=["speaker_id", "duration"], batch_size=batch_size, use_threads=False):
        for speaker, duration in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
            rows += 1
            try:
                speaker_id = str(speaker).strip()
                seconds = float(duration)
            except (TypeError, ValueError):
                invalid += 1
                continue
            if not speaker_id or not math.isfinite(seconds) or seconds <= 0:
                invalid += 1
                continue
            samples, frames = count_frames(seconds)
            total = totals[speaker_id]
            total[0] += 1
            total[1] += frames
            total[2] += samples
    return dict(totals), rows, invalid


def main(args: argparse.Namespace) -> None:
    if min(args.min_frames, args.rank, args.workers, args.batch_size) < 1:
        raise ValueError("all numeric arguments must be positive")
    shards = sorted(args.parquet_root.resolve().glob("train-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No train Parquet shards under {args.parquet_root}")
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    rows = invalid = 0
    with ThreadPoolExecutor(max_workers=min(args.workers, len(shards))) as pool:
        jobs = {pool.submit(scan_shard, shard, args.batch_size): shard for shard in shards}
        for index, future in enumerate(as_completed(jobs), start=1):
            shard_totals, shard_rows, shard_invalid = future.result()
            rows += shard_rows
            invalid += shard_invalid
            for speaker, values in shard_totals.items():
                total = totals[speaker]
                total[0] += values[0]
                total[1] += values[1]
                total[2] += values[2]
            print(f"[{index}/{len(shards)}] {jobs[future].name}: rows={shard_rows} invalid={shard_invalid}", flush=True)
    speakers = [
        {"speaker_id": speaker, "utterances": value[0], "wavlm_frames": value[1],
         "seconds_at_16k": value[2] / SAMPLE_RATE, "minutes_at_16k": value[2] / (60 * SAMPLE_RATE),
         "meets_source_cap": value[1] >= args.min_frames}
        for speaker, value in totals.items()
    ]
    speakers.sort(key=lambda item: (-item["wavlm_frames"], item["speaker_id"]))
    qualified = [item for item in speakers if item["meets_source_cap"]]
    report = {
        "parquet_root": str(args.parquet_root.resolve()), "split": "train", "shards": len(shards), "rows": rows,
        "invalid_rows": invalid, "speakers": len(speakers), "speakers_meeting_source_cap": len(qualified),
        "rank_feasible": len(qualified) >= args.rank,
        "criterion": {"rank": args.rank, "source_frame_cap": args.min_frames, "sample_rate": SAMPLE_RATE,
                      "receptive_field_samples": RECEPTIVE_FIELD, "hop_samples": HOP,
                      "minimum_audio_minutes_approximately": args.min_frames * HOP / (60 * SAMPLE_RATE)},
        "qualifying_speakers": qualified, "top_speakers": speakers[:50],
    }
    atomic_write(args.output_json, json.dumps(report, indent=2, sort_keys=True) + "\n")
    buffer = StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(("speaker_id", "utterances", "wavlm_frames", "seconds_at_16k", "minutes_at_16k", "meets_source_cap"))
    for item in speakers:
        writer.writerow((item["speaker_id"], item["utterances"], item["wavlm_frames"],
                         f"{item['seconds_at_16k']:.3f}", f"{item['minutes_at_16k']:.3f}", item["meets_source_cap"]))
    atomic_write(args.output_csv, buffer.getvalue())
    print(f"rows={rows} invalid={invalid} speakers={len(speakers)} at_least_{args.min_frames}_frames={len(qualified)} rank_{args.rank}_feasible={len(qualified) >= args.rank}", flush=True)


if __name__ == "__main__":
    main(parse_args())
