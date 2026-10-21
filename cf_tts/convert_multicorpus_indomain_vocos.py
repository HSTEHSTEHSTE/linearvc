#!/usr/bin/env python3
"""Render a 4x4 in-domain multicorpus UTXSS voice-conversion grid with Vocos.

One enrolled speaker is selected from each of LibriSpeech, MSP-Podcast,
Mixer6, and GLOBE.  Five deterministic short clips per source speaker are
converted to each of the four enrolled targets, including identity directions:
4 sources x 5 clips x 4 targets = 80 WAVs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
import yaml

TARGET_SAMPLE_RATE = 16_000
STREAM_CHUNK_FRAMES = 10
CORPORA = ("librispeech", "msp", "mixer6", "globe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factorization-root", type=Path, required=True)
    parser.add_argument("--vocos-project", type=Path, required=True)
    parser.add_argument("--vocos-config", type=Path, required=True)
    parser.add_argument("--vocos-checkpoint", type=Path, required=True)
    parser.add_argument("--inputs-manifest", type=Path, required=True)
    parser.add_argument("--causal-features-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--clips-per-source", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.clips_per_source < 1:
        parser.error("--clips-per-source must be positive")
    return args


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(value)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_vocos(project: Path, config_path: Path, checkpoint_path: Path, device: torch.device):
    sys.path.insert(0, str(project))
    from infer_wavlm_causal import decode_precomputed_features_streaming, instantiate_model

    config = yaml.safe_load(config_path.read_text())
    data_config = config["data"]["init_args"]["val_params"]
    model_config = config["model"]["init_args"]
    content_dim = int(data_config["feature_dim"])
    lifted_dim = int(model_config["feature_extractor"]["init_args"]["feature_dim"])
    sample_rate = int(model_config["sample_rate"])
    if (
        content_dim != 75
        or lifted_dim != 1024
        or not bool(data_config["use_speaker_transform"])
        or int(data_config["speaker_transform_output_dim"]) != 1024
        or sample_rate != TARGET_SAMPLE_RATE
    ):
        raise ValueError(
            "Expected the 16-kHz KV-distilled-L6 content75/lift1024 Vocos configuration; got "
            f"content_dim={content_dim}, lifted_dim={lifted_dim}, "
            f"use_speaker_transform={data_config['use_speaker_transform']}, sample_rate={sample_rate}"
        )
    return instantiate_model(config, checkpoint_path, device), decode_precomputed_features_streaming


def main(args: argparse.Namespace) -> None:
    root = args.factorization_root.resolve()
    for path in (root / "UTXSS.npy", root / "transforms.npy", root / "selection.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This conversion job requires a CUDA GPU")
    staged = json.loads(args.inputs_manifest.read_text())
    causal_features = json.loads(args.causal_features_manifest.read_text())
    speakers = {corpus: str(staged["speakers"][corpus]) for corpus in CORPORA}
    records = {corpus: staged["sources"][corpus] for corpus in CORPORA}
    feature_records = {
        (str(item["source_corpus"]), int(item["clip_index"])): item
        for item in causal_features["records"]
    }
    if int(staged["clips_per_source"]) != args.clips_per_source:
        raise ValueError("--clips-per-source does not match the staged source manifest")
    transforms = np.load(root / "transforms.npy", allow_pickle=True).item()
    if not isinstance(transforms, dict):
        raise TypeError("transforms.npy is not a packed transform dictionary")
    source_ids = list(speakers.values())
    missing = [speaker for speaker in source_ids if speaker not in transforms]
    if missing:
        raise KeyError(f"Selected speakers missing from transforms.npy: {missing}")
    projection = torch.from_numpy(np.load(root / "UTXSS.npy")).to(device=device, dtype=torch.float32)
    if tuple(projection.shape) != (1024, 75):
        raise ValueError(f"Expected UTXSS (1024, 75), got {tuple(projection.shape)}")
    target_transforms = {
        corpus: torch.from_numpy(np.asarray(transforms[speaker])).to(device=device, dtype=torch.float32)
        for corpus, speaker in speakers.items()
    }
    if any(tuple(value.shape) != (75, 1024) for value in target_transforms.values()):
        raise ValueError("All selected target transforms must have shape (75, 1024)")

    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_text(
        args.output_root / "selection.json",
        json.dumps(
            {
                "sources_and_targets": speakers,
                "clips_per_source": args.clips_per_source,
                "directions": len(speakers) ** 2,
                "expected_wavs": len(speakers) ** 2 * args.clips_per_source,
                "factorization_root": str(root),
                "vocos_config": str(args.vocos_config),
                "vocos_checkpoint": str(args.vocos_checkpoint),
                "causal_features_manifest": str(args.causal_features_manifest),
                "feature_regime": "KV-distilled WavLM-L6, 20-ms pushes, 10-s history, UTXSS-75, target lift-1024",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    print(f"Selected speakers: {speakers}", flush=True)
    vocos, decode_streaming = load_vocos(args.vocos_project, args.vocos_config, args.vocos_checkpoint, device)
    manifest: list[dict[str, object]] = []
    rendered = 0
    for source_corpus in CORPORA:
        source_speaker = speakers[source_corpus]
        for clip_index, staged_record in enumerate(records[source_corpus], start=1):
            record = staged_record["source_record"]
            feature_record = feature_records.get((source_corpus, clip_index))
            if feature_record is None:
                raise KeyError(f"Missing KV-distilled features for {source_corpus}:{clip_index}")
            feature_path = Path(feature_record["feature_path"])
            try:
                features = torch.load(feature_path, map_location=device, weights_only=True)
            except TypeError:
                features = torch.load(feature_path, map_location=device)
            if not isinstance(features, torch.Tensor) or tuple(features.shape[1:]) != (1024,) or features.shape[0] < 1:
                raise ValueError(f"Expected nonempty [frames, 1024] KV-distilled tensor: {feature_path}")
            features = features.float()
            content = features @ projection
            for target_corpus in CORPORA:
                target_speaker = speakers[target_corpus]
                converted = content @ target_transforms[target_corpus]
                with torch.inference_mode():
                    prediction = decode_streaming(
                        vocos,
                        converted.transpose(0, 1).unsqueeze(0),
                        STREAM_CHUNK_FRAMES,
                    )[0].float().cpu().clamp(-1, 1)
                output_dir = args.output_root / f"{source_corpus}_to_{target_corpus}"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{clip_index:02d}_{record['id']}.wav"
                torchaudio.save(output_path, prediction.unsqueeze(0), TARGET_SAMPLE_RATE)
                manifest.append(
                    {
                        "source_corpus": source_corpus,
                        "source_speaker": source_speaker,
                        "target_corpus": target_corpus,
                        "target_speaker": target_speaker,
                        "source_record": record,
                        "kv_distilled_feature": str(feature_path),
                        "output": str(output_path),
                        "sample_rate": TARGET_SAMPLE_RATE,
                        "output_samples": prediction.numel(),
                    }
                )
                rendered += 1
                print(
                    f"[{rendered}/{len(CORPORA) ** 2 * args.clips_per_source}] "
                    f"{source_corpus}:{clip_index} -> {target_corpus}",
                    flush=True,
                )
            del features, content
    atomic_text(args.output_root / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(f"CONVERSION_COMPLETED outputs={rendered} root={args.output_root}", flush=True)


if __name__ == "__main__":
    main(parse_args())
