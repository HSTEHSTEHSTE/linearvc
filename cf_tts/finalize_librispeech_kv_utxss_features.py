#!/usr/bin/env python3
"""Verify KV-cached LibriSpeech 75-D extraction and write provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--projection-path", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    return parser.parse_args()


def atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    feature_root = args.feature_root.expanduser().resolve()
    metadata_root = args.metadata_root.expanduser().resolve()
    projection_path = args.projection_path.expanduser().resolve()
    missing: list[str] = []
    expected = 0
    bytes_total = 0
    for shard in range(args.shards):
        manifest = metadata_root / "manifests" / f"shard_{shard}_of_{args.shards}.tsv"
        if not manifest.is_file():
            raise SystemExit(f"Missing manifest: {manifest}")
        for line in manifest.read_text().splitlines():
            relative = line.split("\t", 1)[0]
            if not relative:
                continue
            expected += 1
            output = (feature_root / relative).with_suffix(".pt")
            if not output.is_file() or output.stat().st_size == 0:
                missing.append(relative)
            else:
                bytes_total += output.stat().st_size
    if missing:
        preview = ", ".join(missing[:20])
        raise SystemExit(f"Missing or empty outputs: {len(missing)}; first: {preview}")
    projection = np.load(projection_path, mmap_mode="r")
    if projection.shape != (1024, 75):
        raise SystemExit(f"Unexpected UTXSS shape: {projection.shape}")
    provenance = {
        "created_at_unix": time.time(),
        "input_root": str(input_root),
        "feature_root": str(feature_root),
        "files_verified": expected,
        "feature_bytes": bytes_total,
        "projection_path": str(projection_path),
        "projection_shape": list(projection.shape),
        "projection_sha256": hashlib.sha256(projection_path.read_bytes()).hexdigest(),
        "source": "official WavLM-Large layer 6; anon-baseline KV cache; cache_frames=250; chunk_samples=3200",
        "external_lookahead": "0 frames",
        "intra_block_attention": "bidirectional within each 200 ms push, matching anon_baseline",
        "dtype": "float32",
        "persistent_1024d_features": False,
    }
    atomic_write_json(metadata_root / "extraction-provenance.json", provenance)
    print(json.dumps(provenance, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
