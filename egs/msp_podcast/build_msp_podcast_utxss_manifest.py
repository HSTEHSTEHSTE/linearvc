#!/usr/bin/env python3
"""Publish a rank-75-ready MSP-Podcast manifest for UTXSS factorization.

Only speakers that passed the persistent frame audit are included.  MSP IDs
are namespaced as ``msp_<id>`` so they cannot collide with LibriSpeech or
Mixer6 IDs when XS blocks are later combined.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

import soundfile as sf


WAVLM_SAMPLE_RATE = 16_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def main(args: argparse.Namespace) -> None:
    root = args.corpus_root.resolve()
    labels_path = root / "Labels" / "labels_consensus.csv"
    audio_root = root / "Audios"
    if not labels_path.is_file() or not audio_root.is_dir():
        raise FileNotFoundError("Expected Labels/labels_consensus.csv and Audios/ under --corpus-root")
    audit = json.loads(args.audit_report.read_text())
    qualified = {
        str(item["speaker_id"])
        for item in audit.get("qualifying_speakers", [])
        if item.get("meets_source_cap")
    }
    if not qualified:
        raise ValueError(f"No qualifying speakers in audit report: {args.audit_report}")

    rows: list[dict[str, object]] = []
    skipped: Counter[str] = Counter()
    seen_filenames: dict[str, str] = {}
    with labels_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"FileName", "SpkrID", "Split_Set"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{labels_path} lacks required columns {sorted(required)}")
        for row in reader:
            filename = row["FileName"].strip()
            raw_speaker = row["SpkrID"].strip()
            if raw_speaker not in qualified:
                continue
            if not filename:
                skipped["empty_filename"] += 1
                continue
            previous = seen_filenames.get(filename)
            if previous is not None:
                if previous != raw_speaker:
                    raise ValueError(f"Audio file maps to two speakers: {filename}: {previous}, {raw_speaker}")
                skipped["duplicate_label_row"] += 1
                continue
            seen_filenames[filename] = raw_speaker
            path = audio_root / filename
            if not path.is_file():
                skipped["missing_audio"] += 1
                continue
            try:
                info = sf.info(path)
            except Exception:
                skipped["unreadable_audio"] += 1
                continue
            if info.samplerate != WAVLM_SAMPLE_RATE or info.channels != 1 or info.frames <= 0:
                skipped["unexpected_audio_format"] += 1
                continue
            stem = Path(filename).stem
            rows.append(
                {
                    "id": f"msp_{stem}",
                    "speaker_id": f"msp_{raw_speaker}",
                    "path": str(path),
                    "start_seconds": 0.0,
                    "duration_seconds": info.frames / info.samplerate,
                    "split": row["Split_Set"].strip(),
                }
            )

    if not rows:
        raise RuntimeError("No MSP-Podcast clips were eligible for UTXSS factorization")
    rows.sort(key=lambda item: (str(item["speaker_id"]), str(item["id"])))
    speaker_counts = Counter(str(item["speaker_id"]) for item in rows)
    output = "".join(json.dumps(item, sort_keys=True) + "\n" for item in rows)
    report = {
        "corpus_root": str(root),
        "audit_report": str(args.audit_report),
        "required_sample_rate": WAVLM_SAMPLE_RATE,
        "qualified_raw_speakers": len(qualified),
        "manifest_speakers": len(speaker_counts),
        "clips_written": len(rows),
        "clips_by_speaker": dict(sorted(speaker_counts.items())),
        "skipped_by_reason": dict(sorted(skipped.items())),
        "output": str(args.output),
    }
    atomic_write(args.output, output)
    atomic_write(args.report, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in report if key != "clips_by_speaker"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(parse_args())
