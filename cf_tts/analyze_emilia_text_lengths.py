"""Report post-phonemization text-length statistics for one Emilia tar shard."""

import argparse
import json
import math
import os
import re
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml
from huggingface_hub import get_token, hf_hub_download

from linearvc.cf_tts.dataset import TTS_Collate
from linearvc.cf_tts.utils.artifacts import configure_artifact_environment


def percentile(values: list[int], fraction: float) -> int:
    """Return a nearest-rank percentile from a non-empty sorted list."""
    index = max(0, min(len(values) - 1, math.ceil(fraction * len(values)) - 1))
    return values[index]


def phone_token_count(collate: TTS_Collate, raw_text: str) -> int | None:
    """Mirror the CF-TTS phone-token conversion without loading audio."""
    text = re.sub(r"\s+", " ", raw_text).strip()
    if not text:
        return None
    if collate.pre_phonemized:
        return sum(phone in collate.text_tokenizer for phone in text.split())

    phonemized = collate.phonemize([text], separator=collate.separator, strip=True)
    if len(phonemized) != 1 or not phonemized[0]:
        return None
    phone = re.sub(r"([;:,.!?¡¿—…\"«»“”\(\)\{\}\[\]])", r"-\1", phonemized[0])
    tokens = phone.replace("- ", "-").replace(" ", "-").split("-")
    return sum(token in collate.text_tokenizer for token in tokens)


def histogram(values: list[int]) -> list[tuple[str, int]]:
    bins = ((1, 32), (33, 64), (65, 128), (129, 256), (257, 512),
            (513, 768), (769, 1024), (1025, 1600))
    rows = [(f"{low}-{high}", sum(low <= value <= high for value in values)) for low, high in bins]
    rows.append((">1600", sum(value > 1600 for value in values)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--shard", required=True)
    parser.add_argument("--stage-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    configure_artifact_environment()
    with args.config.open() as stream:
        config = yaml.safe_load(stream)
    emilia = config["data"]["train"]["emilia"]
    token_env = emilia.get("hf_token_env", "HF_TOKEN")
    token = os.environ.get(token_env) or get_token()
    if not token:
        raise RuntimeError(f"No Hugging Face token available in ${token_env} or cached login.")

    args.stage_dir.mkdir(parents=True, exist_ok=True)
    shard_path = Path(hf_hub_download(
        repo_id=emilia["repo_id"],
        filename=args.shard,
        repo_type="dataset",
        revision=emilia.get("revision", "main"),
        token=token,
        local_dir=str(args.stage_dir),
    ))
    collate = TTS_Collate(config)
    text_key = emilia.get("text_key", "text")
    max_tokens = int(config["data"]["train"]["streaming_max_tokens_per_utterance"])

    raw_records = empty_text = malformed_json = unphonemizable = 0
    raw_words: list[int] = []
    phone_lengths: list[int] = []
    with tarfile.open(shard_path, mode="r:*") as archive:
        for member in archive:
            if not member.isfile() or not member.name.lower().endswith(".json"):
                continue
            raw_records += 1
            extracted = archive.extractfile(member)
            if extracted is None:
                malformed_json += 1
                continue
            with extracted:
                try:
                    metadata = json.loads(extracted.read().decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    malformed_json += 1
                    continue
            text = metadata.get(text_key)
            if not isinstance(text, str) or not text.strip():
                empty_text += 1
                continue
            raw_words.append(len(text.split()))
            count = phone_token_count(collate, text)
            if not count:
                unphonemizable += 1
                continue
            phone_lengths.append(count)

    phone_lengths.sort()
    raw_words.sort()
    if not phone_lengths:
        raise RuntimeError("No phonemizable transcripts found in the selected shard.")

    over_limit = sum(length > max_tokens for length in phone_lengths)
    percentiles = (("p50", 0.50), ("p90", 0.90), ("p95", 0.95),
                   ("p99", 0.99), ("p99.5", 0.995), ("p99.9", 0.999))
    report = [
        "# Emilia text-length scan",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Shard: `{args.shard}`",
        f"- Metadata records: {raw_records}",
        f"- Empty text: {empty_text}",
        f"- Malformed JSON: {malformed_json}",
        f"- Unphonemizable text: {unphonemizable}",
        f"- Usable phone-token sequences: {len(phone_lengths)}",
        f"- Current discard threshold: {max_tokens} phone tokens",
        f"- Would be discarded by threshold: {over_limit} ({over_limit / len(phone_lengths):.3%})",
        "",
        "## Phone-token length quantiles",
        "",
        "| Statistic | Tokens |",
        "| --- | ---: |",
        f"| min | {phone_lengths[0]} |",
        *[f"| {name} | {percentile(phone_lengths, fraction)} |" for name, fraction in percentiles],
        f"| max | {phone_lengths[-1]} |",
        "",
        "## Phone-token histogram",
        "",
        "| Token range | Records | Fraction |",
        "| --- | ---: | ---: |",
        *[
            f"| {label} | {count} | {count / len(phone_lengths):.3%} |"
            for label, count in histogram(phone_lengths)
        ],
        "",
        "## Raw whitespace-word lengths",
        "",
        f"- p50: {percentile(raw_words, 0.50)}",
        f"- p95: {percentile(raw_words, 0.95)}",
        f"- p99: {percentile(raw_words, 0.99)}",
        f"- max: {raw_words[-1]}",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(report) + "\n")
    print(args.report)


if __name__ == "__main__":
    main()
