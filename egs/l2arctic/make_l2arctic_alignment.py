#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
import re


def parse_textgrid(textgrid_path):
    """
    Parse a Praat TextGrid file into a dictionary of tiers.

    Output format:
    {
        "words": [
            {"start": 0.03, "end": 0.33, "text": "for"},
            ...
        ],
        "phones": [
            {"start": 0.03, "end": 0.17, "text": "F"},
            ...
        ],
        "IPA": [
            {"start": 4.65, "end": 4.71, "text": "d,t,s"},
            ...
        ]
    }
    """

    textgrid_path = Path(textgrid_path)

    with textgrid_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    tiers = {}

    current_tier_name = None
    inside_interval = False
    current_interval = {}

    tier_name_re = re.compile(r'^name\s*=\s*"(.*)"\s*$')
    xmin_re = re.compile(r'^xmin\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*$')
    xmax_re = re.compile(r'^xmax\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*$')
    text_re = re.compile(r'^text\s*=\s*"(.*)"\s*$')

    for raw_line in lines:
        line = raw_line.strip()

        tier_name_match = tier_name_re.match(line)
        if tier_name_match:
            current_tier_name = tier_name_match.group(1)
            tiers[current_tier_name] = []
            continue

        if line.startswith("intervals ["):
            inside_interval = True
            current_interval = {}
            continue

        if inside_interval and current_tier_name is not None:
            xmin_match = xmin_re.match(line)
            xmax_match = xmax_re.match(line)
            text_match = text_re.match(line)

            if xmin_match:
                current_interval["start"] = float(xmin_match.group(1))
                continue

            if xmax_match:
                current_interval["end"] = float(xmax_match.group(1))
                continue

            if text_match:
                current_interval["text"] = text_match.group(1)

                if (
                    "start" in current_interval
                    and "end" in current_interval
                    and "text" in current_interval
                ):
                    tiers[current_tier_name].append(current_interval)

                inside_interval = False
                current_interval = {}
                continue

    return tiers


def is_in_textgrid_folder(tg_path):
    """
    Return True only if the TextGrid file is inside a folder named 'textgrid',
    case-insensitive.

    This excludes files inside 'annotation' folders.
    """

    return any(part.lower() == "textgrid" for part in Path(tg_path).parts)


def infer_speaker_from_path(tg_path, textgrid_root):
    """
    Infer speaker ID from the TextGrid path.

    Expected layout:

        L2ARCTIC/SPEAKER/textgrid/arctic_a0001.TextGrid

    With textgrid_root = L2ARCTIC, this returns SPEAKER.
    """

    tg_path = Path(tg_path)
    textgrid_root = Path(textgrid_root)

    relative_path = tg_path.relative_to(textgrid_root)

    if len(relative_path.parts) < 3:
        raise ValueError(
            f"Could not infer speaker from path: {tg_path}\n"
            f"Expected file to be under: {textgrid_root}/SPEAKER/textgrid/"
        )

    speaker = relative_path.parts[0]

    return speaker


def build_alignments(
    textgrid_root,
    include_empty=True,
    include_silence=True,
    allowed_tiers=None,
):
    """
    Recursively collect only TextGrid files inside folders named 'textgrid'
    under textgrid_root.

    Files inside 'annotation' folders are ignored.

    Output format:
    {
        "ABA": {
            "arctic_a0001": {
                "words": [...],
                "phones": [...],
                "IPA": [...]
            },
            ...
        },
        "BWC": {
            "arctic_a0001": {
                "words": [...],
                "phones": [...],
                "IPA": [...]
            },
            ...
        }
    }
    """

    textgrid_root = Path(textgrid_root)

    if not textgrid_root.exists():
        raise FileNotFoundError(f"TextGrid root does not exist: {textgrid_root}")

    all_textgrid_files = sorted(
        list(textgrid_root.rglob("*.TextGrid"))
        + list(textgrid_root.rglob("*.textgrid"))
    )

    textgrid_files = [
        tg_path
        for tg_path in all_textgrid_files
        if is_in_textgrid_folder(tg_path)
    ]

    alignments = {}

    for tg_path in textgrid_files:
        speaker = infer_speaker_from_path(tg_path, textgrid_root)
        stem = tg_path.stem

        parsed = parse_textgrid(tg_path)

        if allowed_tiers is not None:
            parsed = {
                tier_name: intervals
                for tier_name, intervals in parsed.items()
                if tier_name in allowed_tiers
            }

        cleaned = {}

        for tier_name, intervals in parsed.items():
            new_intervals = []

            for interval in intervals:
                label = interval.get("text", "")

                if not include_empty and label == "":
                    continue

                if not include_silence and label.strip() in {"sil", "sp"}:
                    continue

                new_intervals.append(interval)

            cleaned[tier_name] = new_intervals

        if speaker not in alignments:
            alignments[speaker] = {}

        if stem in alignments[speaker]:
            raise ValueError(
                f"Duplicate TextGrid stem found for speaker '{speaker}': {stem}\n"
                f"Existing key would be overwritten by: {tg_path}"
            )

        alignments[speaker][stem] = cleaned

    return alignments


def save_json(data, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Consolidate L2-ARCTIC TextGrid alignments into one alignments.json "
            "keyed by speaker, then wav/TextGrid stem. Only TextGrid files inside "
            "folders named 'textgrid' are used."
        )
    )

    parser.add_argument(
        "textgrid_root",
        type=str,
        help="Root directory containing speaker folders with textgrid folders, e.g. L2ARCTIC/",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="alignments.json",
        help="Output JSON path. Default: alignments.json",
    )

    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help='Skip intervals where text == "".',
    )

    parser.add_argument(
        "--skip-silence",
        action="store_true",
        help='Skip silence/pause intervals where text is "sil" or "sp".',
    )

    parser.add_argument(
        "--tiers",
        nargs="+",
        default=None,
        help=(
            "Optional list of tiers to keep. "
            'Example: --tiers words phones IPA'
        ),
    )

    args = parser.parse_args()

    allowed_tiers = set(args.tiers) if args.tiers is not None else None

    alignments = build_alignments(
        textgrid_root=args.textgrid_root,
        include_empty=not args.skip_empty,
        include_silence=not args.skip_silence,
        allowed_tiers=allowed_tiers,
    )

    save_json(alignments, args.output)

    num_speakers = len(alignments)
    num_utterances = sum(len(utts) for utts in alignments.values())

    print(f"Saved {num_utterances} utterance alignments for {num_speakers} speakers to: {args.output}")


if __name__ == "__main__":
    main()