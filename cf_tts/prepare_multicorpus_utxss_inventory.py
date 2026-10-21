#!/usr/bin/env python3
"""Prepare deterministic manifests for a four-corpus no-single-anchor UTXSS fit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
import soundfile as sf


SAMPLE_RATE = 16_000
RECEPTIVE_FIELD = 400
HOP = 320


@dataclass(frozen=True)
class AudioRecord:
    corpus: str
    speaker_id: str
    identifier: str
    payload: dict[str, Any]
    duration_seconds: float
    frames: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--librispeech-root", type=Path, required=True)
    parser.add_argument("--msp-manifest", type=Path, required=True)
    parser.add_argument("--mixer-manifest", type=Path, required=True)
    parser.add_argument("--globe-parquet-root", type=Path, required=True)
    parser.add_argument("--globe-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-speakers-per-corpus", type=int, default=200)
    parser.add_argument("--min-frames", type=int, default=32_768)
    parser.add_argument("--reference-candidates", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def frame_count(seconds: float) -> int:
    samples = round(seconds * SAMPLE_RATE)
    return 0 if samples < RECEPTIVE_FIELD else 1 + (samples - RECEPTIVE_FIELD) // HOP


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def record_hash(record: AudioRecord) -> int:
    return int.from_bytes(hashlib.sha256(f"multicorpus-utxss-v1:{record.identifier}".encode()).digest()[:8], "big")


def choose_reference(records: Iterable[AudioRecord], count: int) -> list[AudioRecord]:
    heap: list[tuple[int, str, AudioRecord]] = []
    for record in records:
        item = (-record_hash(record), record.identifier, record)
        if len(heap) < count:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    return [item[2] for item in sorted(heap, key=lambda item: (-item[0], item[1]))]


def write_jsonl(path: Path, records: Iterable[AudioRecord]) -> None:
    atomic_text(path, "".join(json.dumps(record.payload, sort_keys=True) + "\n" for record in records))


def librispeech_records(root: Path, workers: int) -> list[AudioRecord]:
    paths = sorted(root.rglob("*.flac"))
    if not paths:
        raise FileNotFoundError(f"No LibriSpeech FLACs under {root}")

    def inspect(path: Path) -> AudioRecord:
        speaker = path.stem.split("-", 1)[0]
        info = sf.info(path)
        if info.channels != 1:
            raise ValueError(f"Expected mono LibriSpeech audio: {path}")
        seconds = info.frames / info.samplerate
        payload = {"kind": "audio", "corpus": "librispeech", "speaker_id": f"libri_{speaker}",
                   "id": f"libri_{path.stem}", "path": str(path), "start_seconds": 0.0,
                   "duration_seconds": seconds}
        return AudioRecord("librispeech", payload["speaker_id"], payload["id"], payload, seconds, frame_count(seconds))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(inspect, paths))


def direct_manifest_records(path: Path, corpus: str, prefix: str) -> list[AudioRecord]:
    records: list[AudioRecord] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            try:
                raw_speaker = str(row["speaker_id"])
                seconds = float(row["duration_seconds"])
                identifier = str(row["id"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: invalid manifest row") from error
            if seconds <= 0:
                continue
            speaker_id = raw_speaker if raw_speaker.startswith(f"{prefix}_") else f"{prefix}_{raw_speaker}"
            identifier = identifier if identifier.startswith(f"{prefix}_") else f"{prefix}_{identifier}"
            payload = dict(row)
            payload.update({"kind": "audio", "corpus": corpus, "speaker_id": speaker_id, "id": identifier})
            records.append(AudioRecord(corpus, payload["speaker_id"], payload["id"], payload, seconds, frame_count(seconds)))
    if not records:
        raise ValueError(f"No usable records in {path}")
    return records


def globe_records(root: Path, selected_speakers: set[str] | None = None) -> list[AudioRecord]:
    records: list[AudioRecord] = []
    for shard in sorted(root.glob("train-*.parquet")):
        parquet = pq.ParquetFile(shard)
        if not {"speaker_id", "duration"}.issubset(parquet.schema_arrow.names):
            raise ValueError(f"{shard} lacks speaker_id/duration")
        for group_index in range(parquet.metadata.num_row_groups):
            table = parquet.read_row_group(group_index, columns=["speaker_id", "duration"])
            speakers = table.column("speaker_id").to_pylist()
            durations = table.column("duration").to_pylist()
            for row_offset, (speaker, duration) in enumerate(zip(speakers, durations)):
                raw_speaker = str(speaker).strip()
                if selected_speakers is not None and raw_speaker not in selected_speakers:
                    continue
                seconds = float(duration)
                if not raw_speaker or seconds <= 0:
                    continue
                payload = {"kind": "globe_parquet", "corpus": "globe", "speaker_id": f"globe_{raw_speaker}",
                           "id": f"globe_{shard.stem}_g{group_index}_r{row_offset}", "parquet_path": str(shard),
                           "row_group": group_index, "row_offset": row_offset, "duration_seconds": seconds}
                records.append(AudioRecord("globe", payload["speaker_id"], payload["id"], payload, seconds, frame_count(seconds)))
    return records


def totals(records: Iterable[AudioRecord]) -> dict[str, dict[str, int | str]]:
    output: dict[str, dict[str, int | str]] = {}
    for record in records:
        entry = output.setdefault(record.speaker_id, {"speaker_id": record.speaker_id, "corpus": record.corpus,
                                                       "utterances": 0, "wavlm_frames": 0})
        entry["utterances"] = int(entry["utterances"]) + 1
        entry["wavlm_frames"] = int(entry["wavlm_frames"]) + record.frames
    return output


def main(args: argparse.Namespace) -> None:
    if min(args.top_speakers_per_corpus, args.min_frames, args.reference_candidates, args.workers) < 1:
        raise ValueError("top-speakers-per-corpus, min-frames, reference-candidates, and workers must be positive")
    out = args.output_dir.resolve()
    libri = librispeech_records(args.librispeech_root.resolve(), args.workers)
    msp = direct_manifest_records(args.msp_manifest, "msp", "msp")
    mixer = direct_manifest_records(args.mixer_manifest, "mixer6", "mixer")
    globe_audit = json.loads(args.globe_audit.read_text())
    globe_totals = {
        f"globe_{item['speaker_id']}": {"speaker_id": f"globe_{item['speaker_id']}", "corpus": "globe",
                                         "utterances": item["utterances"], "wavlm_frames": item["wavlm_frames"]}
        for item in globe_audit["qualifying_speakers"]
    }
    all_totals: dict[str, dict[str, int | str]] = {}
    for corpus_records in (libri, msp, mixer):
        all_totals.update(totals(corpus_records))
    all_totals.update(globe_totals)
    eligible = [entry for entry in all_totals.values() if int(entry["wavlm_frames"]) >= args.min_frames]
    eligible_by_corpus: dict[str, list[dict[str, int | str]]] = defaultdict(list)
    for entry in eligible:
        eligible_by_corpus[str(entry["corpus"])].append(entry)
    selected = []
    selected_by_corpus: dict[str, int] = {}
    for corpus in ("librispeech", "msp", "mixer6", "globe"):
        ranked = sorted(eligible_by_corpus[corpus], key=lambda item: (-int(item["wavlm_frames"]), str(item["speaker_id"])))
        selected.extend(ranked[:args.top_speakers_per_corpus])
        selected_by_corpus[corpus] = min(len(ranked), args.top_speakers_per_corpus)
    selected.sort(key=lambda item: (str(item["corpus"]), -int(item["wavlm_frames"]), str(item["speaker_id"])))
    selected_ids = {str(item["speaker_id"]) for item in selected}
    selected_raw_globe = {speaker.removeprefix("globe_") for speaker in selected_ids if speaker.startswith("globe_")}
    print(f"eligible={len(eligible)} selected={len(selected)} by_corpus={selected_by_corpus}", flush=True)

    direct_targets = [record for record in [*libri, *msp, *mixer] if record.speaker_id in selected_ids]
    globe_targets = globe_records(args.globe_parquet_root.resolve(), selected_raw_globe) if selected_raw_globe else []
    selected_counts = totals([*direct_targets, *globe_targets])
    missing = selected_ids.difference(selected_counts)
    if missing:
        raise RuntimeError(f"Selected speakers had no materialized target records: {sorted(missing)[:10]}")

    references = {
        "librispeech": choose_reference(libri, args.reference_candidates),
        "msp": choose_reference(msp, args.reference_candidates),
        "mixer6": choose_reference(mixer, args.reference_candidates),
        "globe": choose_reference(globe_records(args.globe_parquet_root.resolve()), args.reference_candidates),
    }
    for corpus, records in references.items():
        if sum(record.frames for record in records) < 8_192:
            raise RuntimeError(f"Reference candidate pool for {corpus} has fewer than 8,192 WavLM frames")
        write_jsonl(out / "references" / f"{corpus}.jsonl", records)
    direct_targets.sort(key=lambda record: (record.speaker_id, record.identifier))
    globe_targets.sort(key=lambda record: (record.speaker_id, record.identifier))
    write_jsonl(out / "targets_audio.jsonl", direct_targets)
    write_jsonl(out / "targets_globe.jsonl", globe_targets)
    manifest = {
        "version": 1, "rank": 75, "source_frames_total": 32_768, "source_frames_per_corpus": 8_192,
        "min_eligible_frames": args.min_frames, "eligible_speakers": len(eligible),
        "eligible_speakers_by_corpus": {corpus: len(eligible_by_corpus[corpus]) for corpus in ("librispeech", "msp", "mixer6", "globe")},
        "top_speakers_per_corpus": args.top_speakers_per_corpus, "selected_speakers_by_corpus": selected_by_corpus,
        "selected_speakers": selected,
        "selected_speaker_count": len(selected), "selected_target_records": {"audio": len(direct_targets), "globe_parquet": len(globe_targets)},
        "reference_candidates_per_corpus": {key: len(value) for key, value in references.items()},
        "reference_candidate_frames_per_corpus": {key: sum(record.frames for record in value) for key, value in references.items()},
        "inputs": {"librispeech_root": str(args.librispeech_root), "msp_manifest": str(args.msp_manifest),
                   "mixer_manifest": str(args.mixer_manifest), "globe_parquet_root": str(args.globe_parquet_root),
                   "globe_audit": str(args.globe_audit)},
    }
    atomic_text(out / "selection.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(parse_args())
