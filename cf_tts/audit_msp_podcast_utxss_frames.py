#!/usr/bin/env python3
"""Audit MSP-Podcast speaker duration for rank-75 UTXSS factorization.

The UTXSS builder's default source cap is 32,768 WavLM frames.  This audit
uses the labeled Train/Development rows, reads audio headers only, rescales
their duration to 16 kHz when necessary, and counts valid WavLM frontend
frames (400-sample receptive field, 320-sample hop).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf


WAVLM_SAMPLE_RATE = 16_000
WAVLM_RECEPTIVE_FIELD = 400
WAVLM_HOP = 320


@dataclass(frozen=True)
class Record:
    filename: str
    speaker_id: str
    split: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--min-frames", type=int, default=32_768)
    parser.add_argument("--rank", type=int, default=75)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def wavlm_frame_count(samples_at_16k: int) -> int:
    return 0 if samples_at_16k < WAVLM_RECEPTIVE_FIELD else 1 + (samples_at_16k - WAVLM_RECEPTIVE_FIELD) // WAVLM_HOP


def records_from_labels(path: Path) -> list[Record]:
    records: list[Record] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"FileName", "SpkrID", "Split_Set"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} lacks required columns {sorted(required)}")
        for row in reader:
            speaker = row["SpkrID"].strip()
            filename = row["FileName"].strip()
            split = row["Split_Set"].strip()
            if not filename or not speaker or speaker.lower() == "hidden":
                continue
            records.append(Record(filename, speaker, split))
    if not records:
        raise ValueError(f"No labeled MSP-Podcast records found in {path}")
    return records


def header(record: Record, audio_root: Path) -> tuple[Record, int, int, int, str | None]:
    path = audio_root / record.filename
    try:
        info = sf.info(path)
        samples_16k = round(info.frames * WAVLM_SAMPLE_RATE / info.samplerate)
        return record, info.frames, info.samplerate, wavlm_frame_count(samples_16k), None
    except Exception as error:  # report individual bad/missing files without stopping the audit
        return record, 0, 0, 0, str(error)


def main() -> None:
    args = parse_args()
    if args.min_frames < 1 or args.rank < 1 or args.workers < 1:
        raise ValueError("min-frames, rank, and workers must be positive")
    root = args.corpus_root.expanduser().resolve()
    labels = root / "Labels" / "labels_consensus.csv"
    audio_root = root / "Audios"
    if not labels.is_file() or not audio_root.is_dir():
        raise FileNotFoundError("Expected Labels/labels_consensus.csv and Audios/ under --corpus-root")
    records = records_from_labels(labels)
    totals: dict[str, dict[str, int]] = defaultdict(lambda: {"utterances": 0, "frames": 0, "samples_16k": 0})
    splits: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    sample_rates: dict[int, int] = defaultdict(int)
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for record, source_samples, sample_rate, frames, error in pool.map(lambda item: header(item, audio_root), records):
            if error is not None:
                failures.append({"filename": record.filename, "speaker_id": record.speaker_id, "error": error})
                continue
            entry = totals[record.speaker_id]
            entry["utterances"] += 1
            entry["frames"] += frames
            entry["samples_16k"] += round(source_samples * WAVLM_SAMPLE_RATE / sample_rate)
            splits[record.speaker_id][record.split] += frames
            sample_rates[sample_rate] += 1
    speakers = [
        {
            "speaker_id": speaker_id,
            "utterances": value["utterances"],
            "wavlm_frames": value["frames"],
            "seconds_at_16k": value["samples_16k"] / WAVLM_SAMPLE_RATE,
            "minutes_at_16k": value["samples_16k"] / (60 * WAVLM_SAMPLE_RATE),
            "meets_source_cap": value["frames"] >= args.min_frames,
            "frames_by_split": dict(sorted(splits[speaker_id].items())),
        }
        for speaker_id, value in totals.items()
    ]
    speakers.sort(key=lambda item: (-item["wavlm_frames"], item["speaker_id"]))
    qualifying = [item for item in speakers if item["meets_source_cap"]]
    report = {
        "corpus_root": str(root),
        "labels": str(labels),
        "criterion": {
            "rank": args.rank,
            "source_frame_cap": args.min_frames,
            "sample_rate": WAVLM_SAMPLE_RATE,
            "receptive_field_samples": WAVLM_RECEPTIVE_FIELD,
            "hop_samples": WAVLM_HOP,
            "minimum_audio_minutes_approximately": args.min_frames * WAVLM_HOP / (60 * WAVLM_SAMPLE_RATE),
        },
        "records_labeled": len(records),
        "records_read": len(records) - len(failures),
        "records_failed": len(failures),
        "speakers_labeled": len(speakers),
        "speakers_meeting_source_cap": len(qualifying),
        "rank_feasible": len(speakers) >= args.rank and bool(qualifying),
        "sample_rates": dict(sorted(sample_rates.items())),
        "qualifying_speakers": qualifying,
        "top_speakers": speakers[:50],
        "failures": failures[:100],
    }
    atomic_text(args.output_json, json.dumps(report, indent=2, sort_keys=True) + "\n")
    csv_buffer = io.StringIO(newline="")
    writer = csv.writer(csv_buffer)
    writer.writerow(("speaker_id", "utterances", "wavlm_frames", "seconds_at_16k", "minutes_at_16k", "meets_source_cap", "frames_by_split"))
    for item in speakers:
        writer.writerow(
            (
                item["speaker_id"],
                item["utterances"],
                item["wavlm_frames"],
                f"{item['seconds_at_16k']:.3f}",
                f"{item['minutes_at_16k']:.3f}",
                item["meets_source_cap"],
                json.dumps(item["frames_by_split"], sort_keys=True),
            )
        )
    atomic_text(args.output_csv, csv_buffer.getvalue())
    print(
        f"records={report['records_read']}/{report['records_labeled']} speakers={report['speakers_labeled']} "
        f"at_least_{args.min_frames}_frames={report['speakers_meeting_source_cap']} "
        f"rank_{args.rank}_feasible={report['rank_feasible']}",
        flush=True,
    )
    for item in qualifying[:20]:
        print(
            f"speaker={item['speaker_id']} utterances={item['utterances']} "
            f"frames={item['wavlm_frames']} minutes={item['minutes_at_16k']:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
