#!/usr/bin/env python
"""Prepare a combined XS directory for all LibriSpeech and Mixer 6 speakers.

Existing aligned blocks are reused as symbolic links. A JSONL manifest is
written only for the LibriSpeech speakers absent from the historical
train-clean-100 factorization, so they can be extracted/aligned separately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def check_argv():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--librispeech-dir", type=Path, required=True)
    parser.add_argument("--librispeech-factorization-dir", type=Path, required=True)
    parser.add_argument("--mixer-factorization-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anchor-speaker", default="103")
    return parser.parse_args()


def load_speakers(path: Path):
    return [str(speaker) for speaker in np.load(path / "used_speakers.npy", allow_pickle=True)]


def link_block(source: Path, destination: Path):
    if destination.is_symlink() or destination.exists():
        if destination.resolve() != source.resolve():
            raise FileExistsError(f"Refusing to replace {destination}; expected {source}")
        return
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.symlink_to(source)


def main(args):
    output_dir = args.output_dir
    xs_dir = output_dir / "XS"
    xs_dir.mkdir(parents=True, exist_ok=True)

    libri_speakers = sorted(
        (path.name for path in args.librispeech_dir.iterdir() if path.is_dir())
    )
    if not libri_speakers or libri_speakers[0] != args.anchor_speaker:
        raise ValueError(
            f"Expected {args.anchor_speaker} to be the first LibriSpeech speaker, got {libri_speakers[:3]}"
        )
    historical_libri_speakers = load_speakers(args.librispeech_factorization_dir)
    mixer_speakers = load_speakers(args.mixer_factorization_dir)
    if historical_libri_speakers[0] != args.anchor_speaker or mixer_speakers[0] != args.anchor_speaker:
        raise ValueError("Both source factorizations must begin with the same anchor")

    historical_libri_set = set(historical_libri_speakers)
    missing_libri_speakers = [speaker for speaker in libri_speakers if speaker not in historical_libri_set]
    mixer_targets = [speaker for speaker in mixer_speakers if speaker != args.anchor_speaker]
    if set(libri_speakers).intersection(mixer_targets):
        raise ValueError("LibriSpeech and Mixer 6 target speaker IDs overlap")

    for speaker in historical_libri_speakers:
        link_block(
            args.librispeech_factorization_dir / "XS" / f"{speaker}.npy",
            xs_dir / f"{speaker}.npy",
        )
    for speaker in mixer_targets:
        link_block(
            args.mixer_factorization_dir / "XS" / f"{speaker}.npy",
            xs_dir / f"{speaker}.npy",
        )

    missing_manifest = output_dir / "missing_librispeech_manifest.jsonl"
    with missing_manifest.open("w") as output:
        for speaker in missing_libri_speakers:
            speaker_dir = args.librispeech_dir / speaker
            files = sorted(speaker_dir.rglob("*.flac"))
            if not files:
                raise FileNotFoundError(f"No FLAC files for missing LibriSpeech speaker {speaker}")
            for wav_path in files:
                info = sf.info(wav_path)
                if info.samplerate != 16_000 or info.channels != 1:
                    raise ValueError(f"Expected mono 16 kHz LibriSpeech audio: {wav_path}")
                record = {
                    "speaker_id": speaker,
                    "path": str(wav_path),
                    "start_seconds": 0.0,
                    "duration_seconds": info.frames / info.samplerate,
                }
                output.write(json.dumps(record) + "\n")

    all_speakers = [*libri_speakers, *sorted(mixer_targets)]
    if len(all_speakers) != len(set(all_speakers)):
        raise ValueError("Combined speaker list is not unique")
    speakers_file = output_dir / "all_speakers.txt"
    speakers_file.write_text("\n".join(all_speakers) + "\n")
    metadata = {
        "anchor_speaker": args.anchor_speaker,
        "librispeech_speakers": len(libri_speakers),
        "historical_librispeech_blocks_reused": len(historical_libri_speakers),
        "librispeech_speakers_to_extract": len(missing_libri_speakers),
        "mixer6_speakers": len(mixer_targets),
        "combined_speakers": len(all_speakers),
    }
    (output_dir / "combined_xs_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print(f"Missing-Libri manifest: {missing_manifest}")
    print(f"Factorization speaker order: {speakers_file}")


if __name__ == "__main__":
    main(check_argv())
