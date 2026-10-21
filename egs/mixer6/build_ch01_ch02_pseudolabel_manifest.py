#!/usr/bin/env python3
"""Build a local Mixer6 CH01/CH02 VAD manifest from pseudo-label sidecars.

The Mixer6 corpus copy used in this workspace excludes the ARTS evaluation
set, while the pseudo-label archive contains all sessions.  This builder
therefore publishes only sessions for which both CH01 and CH02 FLACs are
present locally.  Each output record is one diarized VAD turn and has the
schema consumed by ``experiments_mixer6.py``.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import soundfile as sf


CHANNELS = ("CH01", "CH02")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pseudo-label-root", type=Path, required=True)
    parser.add_argument("--mixer6-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content)
    temporary.replace(path)


def load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def local_audio_path(mixer6_root: Path, source_path: str) -> Path:
    marker = "/data/pcm_flac/"
    if marker not in source_path:
        raise ValueError(f"Pseudo-label path lacks {marker!r}: {source_path}")
    return mixer6_root / "data" / "pcm_flac" / source_path.split(marker, 1)[1]


def main(args: argparse.Namespace) -> None:
    pseudo_root = args.pseudo_label_root
    audio_root = args.mixer6_root / "data" / "pcm_flac"
    if not pseudo_root.is_dir():
        raise FileNotFoundError(f"Missing pseudo-label directory: {pseudo_root}")
    if not audio_root.is_dir():
        raise FileNotFoundError(f"Missing Mixer6 FLAC root: {audio_root}")

    records: list[dict[str, Any]] = []
    usable_sessions = 0
    skipped_sessions: Counter[str] = Counter()
    channel_records: Counter[str] = Counter()
    speaker_records: Counter[str] = Counter()
    total_segments = 0

    label_paths = sorted(
        path for path in pseudo_root.glob("*.json") if not path.name.endswith(".audit.json")
    )
    if not label_paths:
        raise FileNotFoundError(f"No pseudo-label JSON files found in {pseudo_root}")

    for labels_path in label_paths:
        session_id = labels_path.stem
        audit_path = labels_path.with_name(f"{session_id}.audit.json")
        if not audit_path.is_file():
            skipped_sessions["missing_audit"] += 1
            continue
        labels = load_json(labels_path)
        audit = load_json(audit_path)
        if not isinstance(labels, list) or not isinstance(audit, dict):
            skipped_sessions["invalid_json_schema"] += 1
            continue
        speaker_channels = audit.get("speaker_channels")
        if not isinstance(speaker_channels, dict) or set(speaker_channels) != set(CHANNELS):
            skipped_sessions["missing_ch01_or_ch02_metadata"] += 1
            continue

        channels: dict[str, tuple[str, Path, float]] = {}
        session_problem: str | None = None
        for channel in CHANNELS:
            metadata = speaker_channels[channel]
            if not isinstance(metadata, dict):
                session_problem = "invalid_channel_metadata"
                break
            speaker = str(metadata.get("speaker", ""))
            source_path = metadata.get("path")
            if not speaker or not isinstance(source_path, str):
                session_problem = "missing_channel_speaker_or_path"
                break
            path = local_audio_path(args.mixer6_root, source_path)
            if not path.is_file():
                session_problem = "missing_local_ch01_or_ch02_audio"
                break
            info = sf.info(path)
            if info.samplerate != 16_000 or info.channels != 1:
                session_problem = "unexpected_audio_format"
                break
            channels[channel] = (speaker, path, info.frames / info.samplerate)
        if session_problem is not None:
            skipped_sessions[session_problem] += 1
            continue

        speaker_to_channel = {speaker: channel for channel, (speaker, _, _) in channels.items()}
        if len(speaker_to_channel) != len(CHANNELS):
            skipped_sessions["same_speaker_on_both_channels"] += 1
            continue

        session_records = 0
        for index, segment in enumerate(labels):
            if not isinstance(segment, dict):
                skipped_sessions["invalid_segment"] += 1
                continue
            try:
                speaker = str(segment["speaker"])
                start = float(segment["start_time"])
                end = float(segment["end_time"])
            except (KeyError, TypeError, ValueError):
                skipped_sessions["invalid_segment"] += 1
                continue
            channel = speaker_to_channel.get(speaker)
            if channel is None:
                skipped_sessions["segment_speaker_not_in_channel_map"] += 1
                continue
            _, path, duration_limit = channels[channel]
            start = max(0.0, start)
            end = min(duration_limit, end)
            if end <= start:
                skipped_sessions["empty_or_out_of_bounds_segment"] += 1
                continue
            records.append(
                {
                    "id": f"{session_id}_{channel}_vad_{index:06d}",
                    "speaker_id": speaker,
                    "path": str(path),
                    "start_seconds": start,
                    "duration_seconds": end - start,
                    "session_id": session_id,
                    "channel": channel,
                    "pseudo_label_file": labels_path.name,
                }
            )
            channel_records[channel] += 1
            speaker_records[speaker] += 1
            session_records += 1
            total_segments += 1
        if session_records:
            usable_sessions += 1
        else:
            skipped_sessions["no_usable_segments"] += 1

    if not records:
        raise RuntimeError("No usable CH01/CH02 pseudo-label segments matched local Mixer6 audio")
    records.sort(key=lambda record: (record["speaker_id"], record["session_id"], record["channel"], record["start_seconds"]))
    manifest = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    report = {
        "pseudo_label_root": str(pseudo_root),
        "mixer6_root": str(args.mixer6_root),
        "audio_channels_required": list(CHANNELS),
        "label_files_seen": len(label_paths),
        "usable_sessions": usable_sessions,
        "skipped_sessions_by_reason": dict(sorted(skipped_sessions.items())),
        "segments_written": total_segments,
        "speaker_count": len(speaker_records),
        "segments_by_channel": dict(sorted(channel_records.items())),
        "output": str(args.output),
    }
    atomic_write(args.output, manifest)
    atomic_write(args.report, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(parse_args())
