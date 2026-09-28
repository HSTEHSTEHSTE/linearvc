#!/usr/bin/env python3
import argparse
import json
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---- helpers ----
STRESS_RE = re.compile(r"\d$")

# CMUdict vowel set in ARPAbet (stress digit may appear in alignments)
VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
    "IH", "IY", "OW", "OY", "UH", "UW"
}

# adjust to your alignment inventory
PAUSE_LABELS = {"SIL", "SP", "SPN", "<sil>", "PAU"}


def unstress(label: str) -> str:
    # IH1 -> IH, AH0 -> AH, ER2 -> ER
    return STRESS_RE.sub("", str(label))


def speaker_from_key(key: str) -> str:
    # LibriSpeech: speaker-chapter-utterance, e.g. 1846-144452-0037
    return key.split("-")[0]


def is_vowel(ph: str) -> bool:
    return unstress(ph) in VOWELS


def is_pause(ph: str) -> bool:
    u = unstress(ph)
    return (ph in PAUSE_LABELS) or (u in PAUSE_LABELS) or (u.lower() == "spn") or (str(ph).lower() == "spn")


def summarize(durs):
    a = np.asarray(durs, dtype=float)
    if a.size == 0:
        return float("nan"), float("nan"), 0
    return float(a.mean()), float(a.var(ddof=0)), int(a.size)


def sorted_ali(ali):
    # Ensure time order, meow
    return sorted(ali, key=lambda z: float(z["start"]))


# ---- main ----
def build_stats(alignments: dict, sil_gap: float = 0.02, eps: float = 1e-8) -> pd.DataFrame:
    """
    Meow: gap handling
      - If gap = next.start - prev.end > sil_gap, treat as a pause gap.
      - For articulation rate normalization, pause gaps are excluded from speaking time.
      - For duration stats, insert a synthetic "spn" segment with dur=gap, so diphone context breaks.
    """
    # normalized phone durations
    all_durs = {}                 # spk -> [dur_norm]
    by_phone = {}                 # spk -> phone -> [dur_norm]
    by_unstress = {}              # spk -> phone_unstress -> [dur_norm]
    by_phone_prev = {}            # spk -> (phone, prev_phone) -> [dur_norm]

    # articulation-rate samples per speaker (one per utterance)
    # articulation rate here = syllables per second over speaking time (pauses removed)
    ar_by_spk = {}                # spk -> [artic_rate_syll_per_sec]

    for utt_id, ali in tqdm(alignments.items(), desc="utt loop"):
        spk = speaker_from_key(utt_id)
        all_durs.setdefault(spk, [])
        by_phone.setdefault(spk, {})
        by_unstress.setdefault(spk, {})
        by_phone_prev.setdefault(spk, {})
        ar_by_spk.setdefault(spk, [])

        ali2 = sorted_ali(ali)

        # --- per-utterance articulation quantities ---
        speak_time = 0.0
        n_syll = 0  # approximate syllable count by counting vowel nuclei

        prev_end = None
        for seg in ali2:
            ph = seg["phoneme"]
            start = float(seg["start"])
            end = float(seg["end"])
            dur = end - start
            if dur <= 0:
                prev_end = max(prev_end, end) if prev_end is not None else end
                continue

            # meow: treat large timestamp gaps as pause gaps (excluded from speak_time)
            if prev_end is not None:
                gap = start - prev_end
                if gap > float(sil_gap):
                    # gap exists, but it's not speaking time; do nothing besides advance prev_end handling
                    pass

            if not is_pause(ph):
                speak_time += dur
                if is_vowel(ph):
                    n_syll += 1

            prev_end = end

        # avg syllable length (sec/syllable) and articulation rate (syll/sec)
        if n_syll > 0 and speak_time > eps:
            avg_syll_len = speak_time / n_syll
            artic_rate = n_syll / speak_time
        else:
            avg_syll_len = None
            artic_rate = None

        if artic_rate is not None:
            ar_by_spk[spk].append(artic_rate)

        # normalization denominator: average syllable length
        norm_denom = max(avg_syll_len, eps) if avg_syll_len is not None else 1.0

        # --- collect normalized durations (with synthetic spn for big gaps) ---
        prev_ph = "<s>"
        prev_end = None

        for seg in ali2:
            ph = seg["phoneme"]
            start = float(seg["start"])
            end = float(seg["end"])
            dur = end - start
            if dur < 0:
                continue

            # meow: insert "spn" segment for big gaps and break context
            if prev_end is not None:
                gap = start - prev_end
                if gap > float(sil_gap):
                    gap_norm = gap / norm_denom
                    all_durs[spk].append(gap_norm)
                    by_phone[spk].setdefault("spn", []).append(gap_norm)
                    by_unstress[spk].setdefault("spn", []).append(gap_norm)
                    by_phone_prev[spk].setdefault(("spn", prev_ph), []).append(gap_norm)
                    prev_ph = "spn"

            dur_norm = dur / norm_denom

            all_durs[spk].append(dur_norm)
            by_phone[spk].setdefault(ph, []).append(dur_norm)
            u = unstress(ph)
            by_unstress[spk].setdefault(u, []).append(dur_norm)
            by_phone_prev[spk].setdefault((ph, prev_ph), []).append(dur_norm)

            prev_ph = ph
            prev_end = end

    rows = []

    # --- articulation rate Gaussian per speaker ---
    for spk, samples in tqdm(ar_by_spk.items(), desc="summarize AR"):
        mean, var, n = summarize(samples)
        rows.append({
            "speaker": spk,
            "stat_type": "articulation_rate_gaussian",  # syll/sec
            "phoneme": None,
            "phoneme_unstressed": None,
            "prev_phoneme": None,
            "n": n,
            "mean": mean,
            "variance": var,
        })

    # --- normalized duration stats per speaker ---
    for spk, durs in tqdm(all_durs.items(), desc="summarize all"):
        mean, var, n = summarize(durs)
        rows.append({
            "speaker": spk,
            "stat_type": "all_phonemes_norm_by_avg_syll_len",
            "phoneme": None,
            "phoneme_unstressed": None,
            "prev_phoneme": None,
            "n": n,
            "mean": mean,
            "variance": var,
        })

    for spk, dct in tqdm(by_phone.items(), desc="summarize by phone"):
        for ph, durs in dct.items():
            mean, var, n = summarize(durs)
            rows.append({
                "speaker": spk,
                "stat_type": "by_phoneme_norm_by_avg_syll_len",
                "phoneme": ph,
                "phoneme_unstressed": None,
                "prev_phoneme": None,
                "n": n,
                "mean": mean,
                "variance": var,
            })

    for spk, dct in tqdm(by_unstress.items(), desc="summarize by unstress"):
        for ph_u, durs in dct.items():
            mean, var, n = summarize(durs)
            rows.append({
                "speaker": spk,
                "stat_type": "by_unstressed_phoneme_norm_by_avg_syll_len",
                "phoneme": None,
                "phoneme_unstressed": ph_u,
                "prev_phoneme": None,
                "n": n,
                "mean": mean,
                "variance": var,
            })

    for spk, dct in tqdm(by_phone_prev.items(), desc="summarize by prev"):
        for (ph, prev_ph), durs in dct.items():
            mean, var, n = summarize(durs)
            rows.append({
                "speaker": spk,
                "stat_type": "by_phoneme_and_prev_norm_by_avg_syll_len",
                "phoneme": ph,
                "phoneme_unstressed": None,
                "prev_phoneme": prev_ph,
                "n": n,
                "mean": mean,
                "variance": var,
            })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alignments_json", type=str,
                    default="/home/hltcoe/xli/ARTS/linearvc/exp/asr/LibriSpeech/alignments/alignments.json")
    ap.add_argument("--out_csv", type=str, default="gt/norm_stats.csv")
    ap.add_argument("--sil_gap", type=float, default=0.02,
                    help="Meow: if next.start - prev.end > sil_gap (sec), treat as a gap and insert synthetic 'spn'.")
    args = ap.parse_args()

    with open(args.alignments_json, "r") as f:
        alignments = json.load(f)

    df = build_stats(alignments, sil_gap=args.sil_gap)
    out_path = args.out_csv
    pd.DataFrame(df).to_csv(out_path, index=False)
    print(f"meow: wrote {out_path}")


if __name__ == "__main__":
    main()