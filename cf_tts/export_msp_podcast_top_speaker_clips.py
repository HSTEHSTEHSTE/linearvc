#!/usr/bin/env python3
"""Export whole MSP-Podcast utterance clips for the top speakers."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import soundfile as sf


TOP_SPEAKERS = ("210", "480", "97", "1383", "2789")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clips-per-speaker", type=int, default=5)
    return parser.parse_args()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def speaker_files(labels: Path) -> dict[str, list[str]]:
    selected: dict[str, list[str]] = defaultdict(list)
    with labels.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            speaker = row.get("SpkrID", "").strip()
            filename = row.get("FileName", "").strip()
            if speaker in TOP_SPEAKERS and filename:
                selected[speaker].append(filename)
    missing = set(TOP_SPEAKERS) - set(selected)
    if missing:
        raise ValueError(f"Top speakers are absent from labels: {sorted(missing)}")
    return selected


def copy_clip(source: Path, destination: Path) -> tuple[int, int]:
    info = sf.info(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return info.samplerate, info.frames


def main() -> None:
    args = parse_args()
    if args.clips_per_speaker < 1:
        raise ValueError("clips-per-speaker must be positive")
    root = args.corpus_root.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to mix with an existing nonempty output directory: {output}")
    audio_root = root / "Audios"
    labels = root / "Labels" / "labels_consensus.csv"
    selected = speaker_files(labels)
    manifest: list[dict[str, object]] = []
    for speaker in TOP_SPEAKERS:
        candidates: list[tuple[float, str]] = []
        for filename in selected[speaker]:
            path = audio_root / filename
            if path.is_file():
                info = sf.info(path)
                candidates.append((info.frames / info.samplerate, filename))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        if len(candidates) < args.clips_per_speaker:
            raise RuntimeError(f"Speaker {speaker} has only {len(candidates)} readable labeled clips")
        for number, (duration, filename) in enumerate(candidates[: args.clips_per_speaker], start=1):
            source = audio_root / filename
            destination = output / f"speaker_{speaker}" / f"{number:02d}_{Path(filename).stem}.wav"
            sample_rate, frames = copy_clip(source, destination)
            manifest.append(
                {
                    "speaker_id": speaker,
                    "source": str(source),
                    "output": str(destination),
                    "source_duration_seconds": duration,
                    "output_duration_seconds": frames / sample_rate,
                    "sample_rate": sample_rate,
                }
            )
            print(f"speaker={speaker} clip={number} source={filename} output={destination.name}", flush=True)
    atomic_json(
        output / "manifest.json",
        {"top_speakers_by_available_wavlm_frames": list(TOP_SPEAKERS), "clips": manifest},
    )
    print(f"Wrote {len(manifest)} excerpts to {output}", flush=True)


if __name__ == "__main__":
    main()
