#!/usr/bin/env python3
"""Run feature-native kNN conversion with the one-frame Vocos decoder.

The source tensor and every target-reference tensor must be rank-2
``(frames, 1024)`` PyTorch tensors. Target tensors are concatenated into the
matching pool; every source frame is replaced by the mean of its ``topk``
cosine-nearest target frames. The pooled one-frame features are decoded by a
local Vocos checkpoint.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
import yaml

from vocos.experiment import VocosExp


FEATURE_DIM = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-feature", type=Path, required=True)
    parser.add_argument(
        "--reference-feature",
        type=Path,
        required=True,
        nargs="+",
        help="One or more target-speaker tensors; their frames form the matching pool.",
    )
    parser.add_argument(
        "--source-audio",
        type=Path,
        required=True,
        help="Original source audio, used only to preserve the Vocos output duration.",
    )
    parser.add_argument("--output-wav", type=Path, required=True)
    parser.add_argument("--vocos-config", type=Path, required=True)
    parser.add_argument("--vocos-checkpoint", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def import_class(class_path: str):
    module_name, class_name = class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def instantiate_vocos(config: dict, checkpoint_path: Path, device: torch.device) -> VocosExp:
    model_config = config["model"]
    model_args = dict(model_config["init_args"])
    for component in ("feature_extractor", "backbone", "head"):
        component_config = model_args[component]
        component_cls = import_class(component_config["class_path"])
        model_args[component] = component_cls(**component_config["init_args"])
    model_cls = import_class(model_config["class_path"])
    model = model_cls(**model_args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()


def load_feature(path: Path, device: torch.device) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        features = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        features = torch.load(path, map_location="cpu")
    if not isinstance(features, torch.Tensor):
        raise TypeError(f"Expected a tensor in {path}, got {type(features)!r}")
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"Expected {path} to have shape (frames, {FEATURE_DIM}), got {tuple(features.shape)}"
        )
    if features.shape[0] == 0:
        raise ValueError(f"Feature tensor has no frames: {path}")
    features = features.float().contiguous()
    if not torch.isfinite(features).all():
        raise ValueError(f"Feature tensor contains NaN or infinity: {path}")
    return features.to(device)


def align_features(features: torch.Tensor, target_frames: int) -> torch.Tensor:
    if features.shape[0] < target_frames:
        features = torch.cat((features, features[-1:].expand(target_frames - features.shape[0], -1)), dim=0)
    return features[:target_frames]


def main() -> None:
    args = parse_args()
    if args.topk < 1:
        raise ValueError("--topk must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    for path in (args.source_audio, args.vocos_config, args.vocos_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_features = load_feature(args.source_feature, device)
    reference_features = [load_feature(path, device) for path in args.reference_feature]
    matching_set = torch.cat(reference_features, dim=0)
    if args.topk > matching_set.shape[0]:
        raise ValueError(
            f"--topk={args.topk} exceeds the {matching_set.shape[0]} reference frames available"
        )

    source_audio, sample_rate = torchaudio.load(args.source_audio)
    if source_audio.shape[0] > 1:
        source_audio = source_audio.mean(dim=0, keepdim=True)
    config = yaml.safe_load(args.vocos_config.read_text())
    data_config = config["data"]["init_args"]["val_params"]
    expected_sample_rate = int(data_config["sampling_rate"])
    feature_hop_length = int(data_config["feature_hop_length"])
    if sample_rate != expected_sample_rate:
        source_audio = torchaudio.functional.resample(source_audio, sample_rate, expected_sample_rate)

    with torch.inference_mode():
        similarities = F.normalize(source_features, dim=-1) @ F.normalize(matching_set, dim=-1).transpose(0, 1)
        neighbour_indices = similarities.topk(args.topk, dim=-1).indices
        pooled_features = matching_set[neighbour_indices].mean(dim=1)
        target_frames = (source_audio.shape[-1] + feature_hop_length - 1) // feature_hop_length
        pooled_features = align_features(pooled_features, target_frames)
        vocos = instantiate_vocos(config, args.vocos_checkpoint, device)
        waveform = vocos(
            source_audio.to(device),
            precomputed_features=pooled_features.transpose(0, 1).unsqueeze(0),
        )[0].float().cpu().clamp(-1, 1)

    args.output_wav.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(args.output_wav), waveform.unsqueeze(0), expected_sample_rate)
    print(
        f"Wrote {args.output_wav} from {source_features.shape[0]} source frames and "
        f"{matching_set.shape[0]} target-reference frames (topk={args.topk}); "
        f"Vocos output frames={pooled_features.shape[0]}."
    )


if __name__ == "__main__":
    main()
