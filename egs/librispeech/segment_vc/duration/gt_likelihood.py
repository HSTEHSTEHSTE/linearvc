#!/usr/bin/env python3
import argparse
import json
import math
import re
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

EPS_VAR = 1e-8
LOG_EPS = 1e-300
STRESS_RE = re.compile(r"(\d)$")

VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
    "IH", "IY", "OW", "OY", "UH", "UW"
}

PAUSE_LABELS = {"SIL", "SP", "SPN", "<sil>", "PAU"}


def canon(ph) -> str:
    s = str(ph).strip()
    if s.lower() == "spn":
        return "spn"
    return s


def unstress(label: str) -> str:
    return STRESS_RE.sub("", canon(label))


def stress_digit(label: str):
    m = STRESS_RE.search(canon(label))
    return int(m.group(1)) if m else None


def is_vowel(ph: str) -> bool:
    return unstress(ph) in VOWELS


def is_pause(ph: str) -> bool:
    ph = canon(ph)
    u = unstress(ph)
    return (ph in PAUSE_LABELS) or (u in PAUSE_LABELS) or (ph.lower() == "spn") or (u.lower() == "spn")


def is_spn(ph: str) -> bool:
    return canon(ph).lower() == "spn" or unstress(ph).lower() == "spn"


def is_start(ph: str) -> bool:
    return canon(ph) == "<s>"


def is_special(ph: str) -> bool:
    return is_spn(ph) or is_start(ph)


def stress_bucket(ph: str) -> str:
    """
    Stress category ONLY (meow):
      - vowel with digit 0 -> V0
      - vowel with digit >=1 -> VS
      - consonant / specials -> NO_STRESS
    """
    ph = canon(ph)
    if is_special(ph):
        return "NO_STRESS"
    d = stress_digit(ph)
    if d is None:
        return "NO_STRESS"
    return "V0" if d == 0 else "VS"


def stress_compatible(query_ph: str, cand_ph: str) -> bool:
    """
    'same stress if applicable' (meow):
      - if query vowel (has digit): require same stress bucket V0 vs VS
      - if query special: must match exactly (<s> with <s>, spn with spn)
      - else consonant: always compatible
    """
    query_ph = canon(query_ph)
    cand_ph = canon(cand_ph)

    if is_special(query_ph):
        return cand_ph == query_ph

    qd = stress_digit(query_ph)
    if qd is None:
        return True

    return stress_bucket(query_ph) == stress_bucket(cand_ph)


def normal_pdf(x, mu, var):
    var = max(float(var), EPS_VAR)
    z = (x - float(mu)) / math.sqrt(var)
    return (1.0 / math.sqrt(2.0 * math.pi * var)) * math.exp(-0.5 * z * z)


def pool_mu_var_from_triples(triples):
    if not triples:
        return None
    n = np.asarray([t[0] for t in triples], dtype=float)
    mu = np.asarray([t[1] for t in triples], dtype=float)
    var = np.asarray([t[2] for t in triples], dtype=float)
    N = n.sum()
    if N <= 0:
        return None
    mu_pool = float((n * mu).sum() / N)
    ex2_pool = float((n * (var + mu * mu)).sum() / N)
    var_pool = max(ex2_pool - mu_pool * mu_pool, 0.0)
    return (mu_pool, float(var_pool), int(N))


def load_phoneme_classes(path: str):
    with open(path, "r") as f:
        obj = json.load(f)

    class_to_phones = {}
    phone_to_classes = defaultdict(set)

    def walk(node, path_parts):
        if isinstance(node, list):
            cname = ".".join(path_parts) if path_parts else "ROOT"
            s = set(str(p) for p in node)
            class_to_phones[cname] = s
            for p in s:
                phone_to_classes[str(p)].add(cname)
            return
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, path_parts + [str(k)])
            return
        raise TypeError(f"Meow: bad node type at {path_parts}: {type(node)}")

    walk(obj, [])

    vowel_set = set()
    for cname, phones in class_to_phones.items():
        if str(cname).startswith("vowels."):
            vowel_set |= set(phones)

    consonant_set = set()
    for cname, phones in class_to_phones.items():
        if str(cname).startswith("vowels."):
            continue
        consonant_set |= set(phones)

    class_to_phones["COARSE.VOWEL"] = vowel_set
    class_to_phones["COARSE.CONSONANT"] = consonant_set
    class_to_phones["COARSE.SPECIAL"] = {"spn", "<s>"}

    for p in vowel_set:
        phone_to_classes[p].add("COARSE.VOWEL")
    for p in consonant_set:
        phone_to_classes[p].add("COARSE.CONSONANT")
    phone_to_classes["spn"].add("COARSE.SPECIAL")
    phone_to_classes["<s>"].add("COARSE.SPECIAL")

    return class_to_phones, phone_to_classes


def fine_class_set(base_phone: str, class_to_phones, phone_to_classes):
    """
    Fine-class relaxation: union of non-COARSE classes only (meow).
    """
    cls = phone_to_classes.get(base_phone, set())
    s = set()
    for c in cls:
        if str(c).startswith("COARSE."):
            continue
        s |= class_to_phones.get(c, set())
    return s


def coarse_tag(ph: str, phone_to_classes) -> str:
    ph = canon(ph)
    if is_special(ph):
        return "COARSE.SPECIAL"
    base = unstress(ph)
    cls = phone_to_classes.get(base, set())
    if "COARSE.VOWEL" in cls:
        return "COARSE.VOWEL"
    if "COARSE.CONSONANT" in cls:
        return "COARSE.CONSONANT"
    return None


def load_gt_models_ref_only(gt_csv: str, speaker_id: str):
    """
    Load reference speaker models from NEW norm_stats.csv:
      - by_phoneme_norm_by_avg_syll_len
      - by_phoneme_and_prev_norm_by_avg_syll_len
      - all_phonemes_norm_by_avg_syll_len
    """
    df = pd.read_csv(gt_csv, low_memory=False)
    df["speaker"] = df["speaker"].astype(str)
    spk_df = df[df["speaker"] == str(speaker_id)]

    # global
    df_all = spk_df[spk_df["stat_type"] == "all_phonemes_norm_by_avg_syll_len"]
    if len(df_all) == 0:
        raise RuntimeError("Meow: missing all_phonemes_norm_by_avg_syll_len for ref speaker.")
    r0 = df_all.iloc[0]
    spk_all = (float(r0["mean"]), float(r0["variance"]))

    # monophones
    spk_phone_exact = {}
    spk_phone_rows = []
    dfp = spk_df[spk_df["stat_type"] == "by_phoneme_norm_by_avg_syll_len"]
    for _, r in dfp.iterrows():
        ph = r.get("phoneme", None)
        if pd.isna(ph):
            continue
        ph = canon(ph)
        n = float(r.get("n", 0.0))
        mu = float(r["mean"])
        var = float(r["variance"])
        spk_phone_exact[ph] = (mu, var)
        spk_phone_rows.append((ph, n, mu, var))

    # diphones
    spk_diphone_exact = {}
    spk_diphone_rows = []
    dfd = spk_df[spk_df["stat_type"] == "by_phoneme_and_prev_norm_by_avg_syll_len"]
    for _, r in dfd.iterrows():
        ph, prev = r.get("phoneme", None), r.get("prev_phoneme", None)
        if pd.isna(ph) or pd.isna(prev):
            continue
        ph = canon(ph)
        prev = canon(prev)
        n = float(r.get("n", 0.0))
        mu = float(r["mean"])
        var = float(r["variance"])
        spk_diphone_exact[(ph, prev)] = (mu, var)
        spk_diphone_rows.append((ph, prev, n, mu, var))

    return spk_all, spk_phone_exact, spk_phone_rows, spk_diphone_exact, spk_diphone_rows


def pick_muvar_phoneme(
    ph,
    spk_all,
    spk_phone_exact,
    spk_phone_rows,
    phone_to_classes,
    class_to_phones,
):
    """
    Monophone backoff (meow):
      1) exact phone
      2) fine class (stress compatible)
      3) coarse class (stress compatible)
      4) global
    """
    ph = canon(ph)

    muvar = spk_phone_exact.get(ph)
    if muvar is not None:
        return float(muvar[0]), float(muvar[1])

    if is_special(ph):
        return float(spk_all[0]), float(spk_all[1])

    ph_u = unstress(ph)

    fine = fine_class_set(ph_u, class_to_phones, phone_to_classes)
    if fine:
        triples = []
        for ph_i, n_i, mu_i, var_i in spk_phone_rows:
            if is_special(ph_i):
                continue
            if unstress(ph_i) in fine and stress_compatible(ph, ph_i):
                triples.append((n_i, mu_i, var_i))
        pooled = pool_mu_var_from_triples(triples)
        if pooled is not None and pooled[2] > 0:
            return float(pooled[0]), float(pooled[1])

    ctag = coarse_tag(ph, phone_to_classes)
    if ctag is not None:
        triples2 = []
        for ph_i, n_i, mu_i, var_i in spk_phone_rows:
            if is_special(ph_i):
                continue
            if coarse_tag(ph_i, phone_to_classes) == ctag and stress_compatible(ph, ph_i):
                triples2.append((n_i, mu_i, var_i))
        pooled2 = pool_mu_var_from_triples(triples2)
        if pooled2 is not None and pooled2[2] > 0:
            return float(pooled2[0]), float(pooled2[1])

    return float(spk_all[0]), float(spk_all[1])


def pick_muvar_diphone(
    ph, prev,
    spk_diphone_exact,
    spk_diphone_rows,
    phone_to_classes,
    class_to_phones,
    backoff_counter=None,
):
    """
    5-step diphone backoff (meow).
    """
    attempted = []

    ph = canon(ph)
    prev = canon(prev)

    ph_u, prev_u = unstress(ph), unstress(prev)
    ph_coarse = coarse_tag(ph, phone_to_classes)
    prev_coarse = coarse_tag(prev, phone_to_classes)

    # 1) exact
    muvar = spk_diphone_exact.get((ph, prev))
    if muvar is not None:
        if backoff_counter is not None:
            backoff_counter["1_exact"] += 1
        return float(muvar[0]), float(muvar[1])
    attempted.append(("1_exact", 0))

    # fine sets (specials are singleton fine sets)
    ph_fine = {ph} if is_special(ph) else fine_class_set(ph_u, class_to_phones, phone_to_classes)
    prev_fine = {prev} if is_special(prev) else fine_class_set(prev_u, class_to_phones, phone_to_classes)

    if not is_special(ph) and not ph_fine:
        raise RuntimeError(f"Meow: no fine class for ph={ph} base={ph_u}")
    if not is_special(prev) and not prev_fine:
        raise RuntimeError(f"Meow: no fine class for prev={prev} base={prev_u}")

    # 2) relax current into fine, prev exact
    triples2 = []
    for ph_i, prev_i, n_i, mu_i, var_i in spk_diphone_rows:
        if prev_i != prev:
            continue
        if (unstress(ph_i) in ph_fine) and stress_compatible(ph, ph_i) and stress_compatible(prev, prev_i):
            triples2.append((n_i, mu_i, var_i))
    attempted.append(("2_curr_fine_prev_exact", len(triples2)))
    pooled2 = pool_mu_var_from_triples(triples2)
    if pooled2 is not None and pooled2[2] > 0:
        if backoff_counter is not None:
            backoff_counter["2_curr_fine_prev_exact"] += 1
        return float(pooled2[0]), float(pooled2[1])

    # 3) relax prev into fine too
    triples3 = []
    for ph_i, prev_i, n_i, mu_i, var_i in spk_diphone_rows:
        if (unstress(ph_i) in ph_fine) and (unstress(prev_i) in prev_fine):
            if stress_compatible(ph, ph_i) and stress_compatible(prev, prev_i):
                triples3.append((n_i, mu_i, var_i))
    attempted.append(("3_curr_fine_prev_fine", len(triples3)))
    pooled3 = pool_mu_var_from_triples(triples3)
    if pooled3 is not None and pooled3[2] > 0:
        if backoff_counter is not None:
            backoff_counter["3_curr_fine_prev_fine"] += 1
        return float(pooled3[0]), float(pooled3[1])

    # 4) relax current into coarse, prev stays fine-relaxed
    triples4 = []
    for ph_i, prev_i, n_i, mu_i, var_i in spk_diphone_rows:
        if coarse_tag(ph_i, phone_to_classes) != ph_coarse:
            continue
        if unstress(prev_i) not in prev_fine:
            continue
        if stress_compatible(ph, ph_i) and stress_compatible(prev, prev_i):
            triples4.append((n_i, mu_i, var_i))
    attempted.append(("4_curr_coarse_prev_fine", len(triples4)))
    pooled4 = pool_mu_var_from_triples(triples4)
    if pooled4 is not None and pooled4[2] > 0:
        if backoff_counter is not None:
            backoff_counter["4_curr_coarse_prev_fine"] += 1
        return float(pooled4[0]), float(pooled4[1])

    # 5) relax prev into coarse too
    triples5 = []
    for ph_i, prev_i, n_i, mu_i, var_i in spk_diphone_rows:
        if coarse_tag(ph_i, phone_to_classes) != ph_coarse:
            continue
        if coarse_tag(prev_i, phone_to_classes) != prev_coarse:
            continue
        if stress_compatible(ph, ph_i) and stress_compatible(prev, prev_i):
            triples5.append((n_i, mu_i, var_i))
    attempted.append(("5_curr_coarse_prev_coarse", len(triples5)))
    pooled5 = pool_mu_var_from_triples(triples5)
    if pooled5 is not None and pooled5[2] > 0:
        if backoff_counter is not None:
            backoff_counter["5_curr_coarse_prev_coarse"] += 1
        return float(pooled5[0]), float(pooled5[1])

    raise KeyError(
        "Meow: no diphone stats found after all requested backoff.\n"
        f"  diphone=(ph={ph}, prev={prev})\n"
        f"  coarse=(ph={ph_coarse}, prev={prev_coarse})\n"
        f"  attempted={attempted}\n"
        f"  ref_rows_loaded={len(spk_diphone_rows)}\n"
    )


def collect_segments_from_alignments_normed(alignments: dict, speaker_id: str, sil_gap: float, eps: float = 1e-8):
    """
    Returns segments with:
      - dur_sec: ground-truth duration in seconds (unnormalized)
      - dur_norm: dur_sec / denom, where denom is avg_syll_len for the utterance
      - denom: per-utterance avg_syll_len (seconds)
    """
    segs = []
    speaker_id = str(speaker_id)

    for utt_id, ali in alignments.items():
        if utt_id.split("-")[0] != speaker_id:
            continue

        speak_time = 0.0
        n_syll = 0
        for seg in ali:
            ph = canon(seg["phoneme"])
            dur = float(seg["end"]) - float(seg["start"])
            if dur <= 0:
                continue
            if not is_pause(ph):
                speak_time += dur
                if is_vowel(ph):
                    n_syll += 1

        denom = (speak_time / n_syll) if (n_syll > 0 and speak_time > eps) else 1.0
        denom = max(denom, eps)

        ali_sorted = sorted(ali, key=lambda z: float(z["start"]))
        prev_label = "<s>"
        prev_end = None

        for seg in ali_sorted:
            ph = canon(seg["phoneme"])
            start = float(seg["start"])
            end = float(seg["end"])
            dur = end - start
            if dur < 0:
                continue

            if prev_end is not None:
                gap = start - prev_end
                if gap > float(sil_gap):
                    segs.append(dict(
                        label="spn",
                        prev_label=canon(prev_label),
                        dur_sec=float(gap),
                        dur_norm=float(gap / denom),
                        denom=float(denom),
                    ))
                    prev_label = "spn"

            segs.append(dict(
                label=ph,
                prev_label=canon(prev_label),
                dur_sec=float(dur),
                dur_norm=float(dur / denom),
                denom=float(denom),
            ))
            prev_label = ph
            prev_end = end

    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alignments_json", type=str,
                    default="/home/hltcoe/xli/ARTS/linearvc/exp/asr/LibriSpeech/alignments/alignments.json")
    ap.add_argument("--gt_stats_csv", type=str, default="gt/norm_stats.csv")
    ap.add_argument("--phoneme_classes_json", type=str, default="phoneme_classes.json")
    ap.add_argument("--src_speaker", type=str, default="61")
    ap.add_argument("--tgt_speaker", type=str, default="121")
    ap.add_argument("--out_csv", type=str, default="loglikelihoods_61_under_121_normed.csv")
    ap.add_argument("--sil_gap", type=float, default=0.02)
    ap.add_argument("--debug_backoff", action="store_true")
    args = ap.parse_args()

    class_to_phones, phone_to_classes = load_phoneme_classes(args.phoneme_classes_json)

    with open(args.alignments_json, "r") as f:
        alignments = json.load(f)

    segs = collect_segments_from_alignments_normed(alignments, args.src_speaker, sil_gap=args.sil_gap)
    if not segs:
        raise RuntimeError("Meow: no segments found for src speaker in alignments.")

    (spk_all,
     spk_phone_exact, spk_phone_rows,
     spk_diphone_exact, spk_diphone_rows) = load_gt_models_ref_only(args.gt_stats_csv, args.tgt_speaker)

    if not spk_phone_rows:
        raise RuntimeError("Meow: no reference phone rows loaded for tgt speaker.")
    if not spk_diphone_rows:
        raise RuntimeError("Meow: no reference diphone rows loaded for tgt speaker.")

    backoff_counter = Counter() if args.debug_backoff else None

    token_ll_phone = []
    per_phone_lls = defaultdict(list)

    token_ll_diphone = []
    per_diphone_lls = defaultdict(list)

    # NEW: unnormalized-duration MSE accumulators (seconds^2), meow
    token_mse_sec_phone = []
    token_mse_sec_diphone = []

    for s in segs:
        x_norm = float(s["dur_norm"])
        x_sec = float(s["dur_sec"])      # ground truth unnormalized duration (seconds)
        denom = float(s["denom"])        # avg syllable length for this utterance (seconds)

        ph = canon(s["label"])
        prev = canon(s["prev_label"])

        # monophone model predicts mu in NORMALIZED units
        mu_norm, var_norm = pick_muvar_phoneme(
            ph,
            spk_all,
            spk_phone_exact,
            spk_phone_rows,
            phone_to_classes,
            class_to_phones,
        )
        ll = math.log(max(normal_pdf(x_norm, mu_norm, var_norm), LOG_EPS))
        token_ll_phone.append(ll)
        per_phone_lls[ph].append(ll)

        # convert predicted mean back to seconds and compute MSE vs ground-truth seconds
        mu_sec = float(mu_norm) * denom
        token_mse_sec_phone.append((x_sec - mu_sec) ** 2)

        # diphone model predicts mu in NORMALIZED units
        mu2_norm, var2_norm = pick_muvar_diphone(
            ph, prev,
            spk_diphone_exact,
            spk_diphone_rows,
            phone_to_classes,
            class_to_phones,
            backoff_counter=backoff_counter,
        )
        ll2 = math.log(max(normal_pdf(x_norm, mu2_norm, var2_norm), LOG_EPS))
        token_ll_diphone.append(ll2)
        per_diphone_lls[(ph, prev)].append(ll2)

        mu2_sec = float(mu2_norm) * denom
        token_mse_sec_diphone.append((x_sec - mu2_sec) ** 2)

    # aggregates (log-likelihoods on normalized durations)
    avg_ll_all_phonemes = float(np.mean(token_ll_phone))
    avg_ll_each_phone = {ph: float(np.mean(v)) for ph, v in per_phone_lls.items()}
    avg_of_phone_avg_lls = float(np.mean(list(avg_ll_each_phone.values()))) if avg_ll_each_phone else np.nan

    avg_ll_all_diphones = float(np.mean(token_ll_diphone))
    avg_ll_each_diphone = {k: float(np.mean(v)) for k, v in per_diphone_lls.items()}
    avg_of_diphone_avg_lls = float(np.mean(list(avg_ll_each_diphone.values()))) if avg_ll_each_diphone else np.nan

    # NEW: global MSE on unnormalized durations (seconds^2), meow
    avg_mse_sec_phone = float(np.mean(token_mse_sec_phone)) if token_mse_sec_phone else np.nan
    avg_mse_sec_diphone = float(np.mean(token_mse_sec_diphone)) if token_mse_sec_diphone else np.nan

    speaker_out = str(args.tgt_speaker)
    rows = []

    # phone outputs
    rows.append(dict(
        speaker=speaker_out, stat_type="loglik_all_phonemes_normed",
        phoneme=None, phoneme_unstressed=None, prev_phoneme=None,
        n=len(token_ll_phone), mean=avg_ll_all_phonemes, variance=np.nan
    ))
    rows.append(dict(
        speaker=speaker_out, stat_type="loglik_by_phoneme_avg_of_avgs_normed",
        phoneme=None, phoneme_unstressed=None, prev_phoneme=None,
        n=len(avg_ll_each_phone), mean=avg_of_phone_avg_lls, variance=np.nan
    ))
    for ph, m in sorted(avg_ll_each_phone.items()):
        rows.append(dict(
            speaker=speaker_out, stat_type="loglik_by_phoneme_normed",
            phoneme=ph, phoneme_unstressed=None, prev_phoneme=None,
            n=len(per_phone_lls[ph]), mean=m, variance=np.nan
        ))

    # diphone outputs
    rows.append(dict(
        speaker=speaker_out, stat_type="loglik_all_diphones_normed",
        phoneme=None, phoneme_unstressed=None, prev_phoneme=None,
        n=len(token_ll_diphone), mean=avg_ll_all_diphones, variance=np.nan
    ))
    rows.append(dict(
        speaker=speaker_out, stat_type="loglik_by_diphone_avg_of_avgs_normed",
        phoneme=None, phoneme_unstressed=None, prev_phoneme=None,
        n=len(avg_ll_each_diphone), mean=avg_of_diphone_avg_lls, variance=np.nan
    ))
    for (ph, prev), m in sorted(avg_ll_each_diphone.items()):
        rows.append(dict(
            speaker=speaker_out, stat_type="loglik_by_diphone_normed",
            phoneme=ph, phoneme_unstressed=None, prev_phoneme=prev,
            n=len(per_diphone_lls[(ph, prev)]), mean=m, variance=np.nan
        ))

    df_out = pd.DataFrame(rows, columns=[
        "speaker", "stat_type", "phoneme", "phoneme_unstressed", "prev_phoneme", "n", "mean", "variance"
    ])
    df_out.to_csv(args.out_csv, index=False)

    print(f"Meow: wrote {args.out_csv}")
    print(f"Meow: segments scored (phones)   n={len(token_ll_phone)} avg_loglik={avg_ll_all_phonemes:.6e}")
    print(f"Meow: segments scored (diphones) n={len(token_ll_diphone)} avg_loglik={avg_ll_all_diphones:.6e}")
    print(f"Meow: avg of per-phone avg loglik   = {avg_of_phone_avg_lls:.6e}")
    print(f"Meow: avg of per-diphone avg loglik = {avg_of_diphone_avg_lls:.6e}")

    # NEW: print unnormalized-duration MSE, meow
    print(f"Meow: MSE (seconds, phones)   n={len(token_mse_sec_phone)} avg_mse={avg_mse_sec_phone:.6e}")
    print(f"Meow: MSE (seconds, diphones) n={len(token_mse_sec_diphone)} avg_mse={avg_mse_sec_diphone:.6e}")

    if backoff_counter is not None:
        print("Meow: diphone backoff usage counts:")
        for k, v in backoff_counter.most_common():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()