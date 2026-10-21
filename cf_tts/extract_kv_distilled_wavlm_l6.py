#!/usr/bin/env python3
"""Extract temporary raw KV-distilled WavLM-L6 features for staged VC clips.

The output is intentionally job-local: it is an intermediate between the
causal WavLM student and Vocos, not a durable feature extraction dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-manifest", type=Path, required=True)
    parser.add_argument("--one-frame-source", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--context-tokens", type=int, default=500)
    parser.add_argument("--chunk-samples", type=int, default=320)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_audio(path: Path) -> torch.Tensor:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != 16_000:
        raise ValueError(f"Expected 16-kHz staged audio, got {sample_rate} Hz: {path}")
    return torch.from_numpy(np.ascontiguousarray(audio.mean(axis=1, dtype=np.float32)))


def main(args: argparse.Namespace) -> None:
    if args.context_tokens < 1 or args.chunk_samples != 320:
        raise ValueError("This deployment extractor requires a 500+ token cache and 20-ms (320-sample) pushes")
    if not torch.cuda.is_available():
        raise RuntimeError("KV-distilled extraction requires CUDA")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must select CUDA")
    if not args.one_frame_source.is_dir() or not args.student_checkpoint.is_file():
        raise FileNotFoundError("Missing one-frame WavLM source tree or student checkpoint")

    sys.path.insert(0, str(args.one_frame_source))
    from causal_wavlm_inference import load_student
    from train import DEFAULT_WAVLM_LARGE_CHECKPOINT

    payload = json.loads(args.inputs_manifest.read_text())
    sources = payload["sources"]
    args.output_root.mkdir(parents=True, exist_ok=True)
    model = load_student(
        args.student_checkpoint,
        Path(DEFAULT_WAVLM_LARGE_CHECKPOINT),
        device,
        torch.float16,
    )
    model.enable_deployment_inference(
        max_context_tokens=args.context_tokens,
        compile_frontend=True,
        compile_ffn=True,
        cuda_graph_ffn=True,
    )
    layers = list(model.student.encoder.layers[: model.output_layer])
    if model.output_layer != 6 or len(layers) != 6 or not all(
        layer.self_attn.use_deployment_fast_path
        and layer.self_attn._packed_qkv_weight is not None
        and layer.self_attn._packed_qkv_bias is not None
        for layer in layers
    ):
        raise RuntimeError("The required optimized six-layer KV-distilled WavLM path was not enabled")

    records: list[dict[str, object]] = []
    total = sum(len(clips) for clips in sources.values())
    complete = 0
    for corpus, clips in sources.items():
        for clip_index, clip in enumerate(clips, start=1):
            audio_path = Path(clip["staged_path"])
            waveform = load_audio(audio_path)
            cache = model.initial_cache()
            chunks: list[torch.Tensor] = []
            with torch.inference_mode():
                for start in range(0, waveform.numel(), args.chunk_samples):
                    chunk = waveform[start : start + args.chunk_samples].unsqueeze(0).to(device, dtype=torch.float16)
                    features, cache = model.forward_chunk(chunk, cache)
                    if features.shape[1]:
                        chunks.append(features.squeeze(0).float().cpu())
            features = torch.cat(chunks, dim=0) if chunks else torch.empty((0, 1024), dtype=torch.float32)
            if features.ndim != 2 or features.shape[1] != 1024 or features.shape[0] == 0:
                raise RuntimeError(f"Unexpected KV-distilled feature shape {tuple(features.shape)} for {audio_path}")
            feature_path = args.output_root / f"{corpus}_{clip_index:02d}.pt"
            torch.save(features, feature_path)
            records.append(
                {
                    "source_corpus": corpus,
                    "clip_index": clip_index,
                    "source_record": clip["source_record"],
                    "staged_path": str(audio_path),
                    "feature_path": str(feature_path),
                    "frames": int(features.shape[0]),
                }
            )
            complete += 1
            print(f"[{complete}/{total}] {corpus}:{clip_index} frames={features.shape[0]}", flush=True)

    triton_calls = [layer.self_attn.deployment_triton_ring_calls for layer in layers]
    if not all(calls > 0 for calls in triton_calls):
        raise RuntimeError(f"The required Triton ring-KV kernel did not run on every layer: {triton_calls}")
    atomic_json(
        args.output_manifest,
        {
            "inputs_manifest": str(args.inputs_manifest),
            "student_checkpoint": str(args.student_checkpoint),
            "context_tokens": args.context_tokens,
            "chunk_samples": args.chunk_samples,
            "deployment": "fused_qkv+triton_ring_attention+compiled_frontend+compiled_cudagraph_ffn",
            "triton_calls_per_layer": triton_calls,
            "records": records,
        },
    )
    print(f"KV_EXTRACTION_COMPLETED clips={complete} manifest={args.output_manifest}", flush=True)


if __name__ == "__main__":
    main(parse_args())
