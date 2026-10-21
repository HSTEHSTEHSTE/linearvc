#!/usr/bin/env python3
"""Stage the 20 short source clips for the multicorpus in-domain VC grid."""

from __future__ import annotations

import argparse
import io
import json
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torchaudio


CORPORA = ("librispeech", "msp", "mixer6", "globe")


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class GlobeAudio:
    """Small row-group cache, kept dependency-light for the staging environment."""

    def __init__(self) -> None:
        self.cache: OrderedDict[tuple[str, int], list[dict[str, object]]] = OrderedDict()

    def read_bytes(self, record: dict[str, object]) -> bytes:
        key = (str(record["parquet_path"]), int(record["row_group"]))
        values = self.cache.get(key)
        if values is None:
            values = pq.ParquetFile(key[0]).read_row_group(key[1], columns=["audio"]).column("audio").to_pylist()
            self.cache[key] = values
            if len(self.cache) > 8:
                self.cache.popitem(last=False)
        audio = values[int(record["row_offset"])]
        if not isinstance(audio, dict) or not isinstance(audio.get("bytes"), bytes):
            raise TypeError(f"GLOBE audio is unavailable for {record['id']}")
        return audio["bytes"]


def read_audio(record: dict[str, object], globe: GlobeAudio) -> np.ndarray:
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
        audio, sample_rate = sf.read(io.BytesIO(globe.read_bytes(record)), dtype="float32", always_2d=True)
        start = round(start_seconds * sample_rate)
        frames = round(duration_seconds * sample_rate)
        audio = audio[start : start + frames]
    else:
        raise ValueError(f"Unsupported audio kind {record['kind']!r}")
    if audio.ndim != 2 or audio.shape[1] != 1:
        raise ValueError(f"Expected mono audio for {record['id']}, got {audio.shape}")
    waveform = torch.from_numpy(np.ascontiguousarray(audio[:, 0]))
    if sample_rate != 16_000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16_000)
    return waveform.numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factorization-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--clips-per-source", type=int, default=5)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    root = args.factorization_root.resolve()
    selection = json.loads((root / "selection.json").read_text())
    speakers = {
        corpus: str(next(item for item in selection["selected_speakers"] if item["corpus"] == corpus)["speaker_id"])
        for corpus in CORPORA
    }
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for path in (root / "targets_audio.jsonl", root / "targets_globe.jsonl"):
        for record in load_jsonl(path):
            if str(record["speaker_id"]) in speakers.values():
                grouped[str(record["speaker_id"])].append(record)
    args.staging_dir.mkdir(parents=True, exist_ok=True)
    globe = GlobeAudio()
    sources: dict[str, list[dict[str, object]]] = {}
    for corpus, speaker in speakers.items():
        candidates = sorted(
            grouped[speaker],
            key=lambda record: (
                not (3.0 <= float(record["duration_seconds"]) <= 10.0),
                float(record["duration_seconds"]),
                str(record["id"]),
            ),
        )
        if len(candidates) < args.clips_per_source:
            raise ValueError(f"{speaker} has only {len(candidates)} clips")
        sources[corpus] = []
        for index, record in enumerate(candidates[: args.clips_per_source], start=1):
            staged_path = args.staging_dir / f"{corpus}_{index:02d}.wav"
            sf.write(staged_path, read_audio(record, globe), 16_000)
            sources[corpus].append({"source_record": record, "staged_path": str(staged_path)})
    args.output_manifest.write_text(
        json.dumps({"speakers": speakers, "clips_per_source": args.clips_per_source, "sources": sources}, indent=2)
        + "\n"
    )
    print(f"STAGING_COMPLETED clips={len(CORPORA) * args.clips_per_source}", flush=True)


if __name__ == "__main__":
    main(parse_args())
