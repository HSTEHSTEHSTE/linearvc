import argparse
from collections import defaultdict
import math
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch, torchaudio
import torch.nn.functional as F
import json
import pandas as pd
import re

STRESS_RE = re.compile(r"\d$")


def unstress(label: str) -> str:
    return STRESS_RE.sub("", str(label))


def phones_in_same_class(phone_u: str, class_to_phones, phone_to_classes):
    """Return set of phones that share at least one class with phone_u."""
    cls = phone_to_classes.get(phone_u, set())
    s = set()
    for c in cls:
        s |= class_to_phones[c]  # each value is already a set[str]
    return s


device = "cuda"

FPS = 50.0
EPS_VAR = 1e-8
LOG_EPS = 1e-300  # floor to avoid log(0), meow


def check_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit_distance_type", type=str,
                        choices=["mean", "dtw", "mean_first_last"], default="mean_first_last")
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--bag_silence", action="store_true")
    parser.add_argument("--segment_splits", type=int, default=1)
    parser.add_argument("--min-segment-length", type=int, default=1)
    parser.add_argument("--target_use_alignment", action="store_true")
    parser.add_argument("--max_segment_length", type=int, default=0)

    parser.add_argument("--w_mean", type=float, default=5.0)
    parser.add_argument("--w_first", type=float, default=1.0)
    parser.add_argument("--w_last", type=float, default=1.0)

    parser.add_argument("--librispeech_root", type=str,
                        default="/home/hltcoe/xli/ARTS/anon_baseline/data/LibriSpeech/test-clean")
    parser.add_argument("--alignments_json", type=str,
                        default="/home/hltcoe/xli/ARTS/linearvc/exp/asr/LibriSpeech/alignments/alignments.json")
    parser.add_argument("--src_speaker", type=str, default="61")
    parser.add_argument("--tgt_speaker", type=str, default="121")

    parser.add_argument("--gt_stats_csv", type=str, default="gt/stats.csv")
    parser.add_argument("--out_csv", type=str, default="stats.csv")
    parser.add_argument("--phoneme_classes_json", type=str, default="phoneme_classes.json")
    return parser.parse_args()


def load_phoneme_classes(path: str):
    """
    Loads nested phoneme class JSON like:
      {"stops":{"voiced":["B","D","G"], ...}, "nasals":["M","N","NG"], ...}

    Returns:
      class_to_phones: dict[str, set[str]]    e.g. "stops.voiced" -> {"B","D","G"}
      phone_to_classes: dict[str, set[str]]   e.g. "G" -> {"stops.voiced"}
    """
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
        raise TypeError(f"Meow: unexpected node type in phoneme_classes.json at {path_parts}: {type(node)}")

    walk(obj, [])

    # ---- SPECIAL token class (<s>, spn) ----
    class_to_phones["SPECIAL.<s>"] = {"<s>", "spn"}
    phone_to_classes["<s>"].add("SPECIAL.<s>")
    phone_to_classes["spn"].add("SPECIAL.<s>")

    # ---- COARSE classes: VOWEL / CONSONANT ----
    vowel_set = set()
    for cname, phones in class_to_phones.items():
        if str(cname).startswith("vowels."):
            vowel_set |= set(phones)

    consonant_set = set()
    for cname, phones in class_to_phones.items():
        cname = str(cname)
        if cname.startswith("vowels.") or cname.startswith("SPECIAL."):
            continue
        consonant_set |= set(phones)

    class_to_phones["COARSE.VOWEL"] = vowel_set
    class_to_phones["COARSE.CONSONANT"] = consonant_set

    for p in vowel_set:
        phone_to_classes[p].add("COARSE.VOWEL")
    for p in consonant_set:
        phone_to_classes[p].add("COARSE.CONSONANT")

    return class_to_phones, phone_to_classes


def length_scaled_pool(X, alpha):
    n = X.shape[0]
    return X.mean(axis=0) * (n ** alpha)


def dtw(X, Y, lam=0.0):
    n, m = len(X), len(Y)
    X_norm = F.normalize(X, p=2, dim=1)
    Y_norm = F.normalize(Y, p=2, dim=1)
    cost_matrix = 1 - X_norm @ Y_norm.T

    D = torch.full((n + 1, m + 1), float("inf"), device=X.device)
    D[0, 0] = 0.0
    lam = torch.tensor(lam, device=X.device)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = cost_matrix[i - 1, j - 1]
            D[i, j] = cost + min(D[i - 1, j - 1], D[i - 1, j] + lam, D[i, j - 1] + lam)
    return D[n, m] / 1000


def split_into_equal_portions(frames, k: int):
    if frames is None or len(frames) == 0:
        return []
    k = max(1, int(k))
    if k == 1:
        return [frames]
    chunks = torch.tensor_split(frames, k, dim=0)
    return [c for c in chunks if c.numel() > 0]


def all_subsegments_leq(x: torch.Tensor, max_len: int):
    T = int(x.shape[0])
    max_len = int(max_len)
    if max_len <= 0 or T <= 0:
        return
    max_len = min(max_len, T)
    for L in range(1, max_len + 1):
        for s in range(0, T - L + 1):
            yield x[s:s + L]


def normal_pdf(x, mu, var):
    var = max(float(var), EPS_VAR)
    z = (x - float(mu)) / math.sqrt(var)
    return (1.0 / math.sqrt(2.0 * math.pi * var)) * math.exp(-0.5 * z * z)


def _pool_mu_var_from_group(df_group: pd.DataFrame):
    n = df_group["n"].astype(float).to_numpy()
    mu = df_group["mean"].astype(float).to_numpy()
    var = df_group["variance"].astype(float).to_numpy()
    N = n.sum()
    if N <= 0:
        return None
    mu_pool = (n * mu).sum() / N
    ex2_pool = (n * (var + mu * mu)).sum() / N
    var_pool = max(ex2_pool - mu_pool * mu_pool, 0.0)
    return float(mu_pool), float(var_pool)


def pool_mu_var_from_triples(triples):
    # triples: list of (n, mean, var)
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


def load_gt_models_with_pooled(gt_csv: str, speaker_id: str):
    df = pd.read_csv(gt_csv)
    df["speaker"] = df["speaker"].astype(str)

    spk_df = df[df["speaker"] == str(speaker_id)]

    spk_all = None
    dfa = spk_df[spk_df["stat_type"] == "all_phonemes"]
    if len(dfa) > 0:
        r = dfa.iloc[0]
        spk_all = (float(r["mean"]), float(r["variance"]))

    # ---------- phones (ref speaker) ----------
    spk_phone = {}
    dfp = spk_df[spk_df["stat_type"] == "by_phoneme"]
    for _, r in dfp.iterrows():
        ph = r["phoneme"]
        if pd.isna(ph):
            continue
        spk_phone[str(ph)] = (float(r["mean"]), float(r["variance"]))

    spk_phone_rows = []  # each: (ph, n, mean, var)
    for _, r in dfp.iterrows():
        ph = r["phoneme"]
        if pd.isna(ph):
            continue
        spk_phone_rows.append((str(ph), float(r["n"]), float(r["mean"]), float(r["variance"])))

    spk_phone_unstress_groups = defaultdict(list)  # u_ph -> [(n,mu,var),...]
    for ph, n, mu, var in spk_phone_rows:
        spk_phone_unstress_groups[unstress(ph)].append((n, mu, var))

    spk_phone_unstress = {}
    for u_ph, triples in spk_phone_unstress_groups.items():
        pooled = pool_mu_var_from_triples(triples)  # (mu,var,N)
        if pooled is not None:
            spk_phone_unstress[u_ph] = pooled

    # ---------- diphones (ref speaker) ----------
    spk_diphone = {}
    dfd = spk_df[spk_df["stat_type"] == "by_phoneme_and_prev"]
    for _, r in dfd.iterrows():
        ph = r["phoneme"]
        prev = r["prev_phoneme"]
        if pd.isna(ph) or pd.isna(prev):
            continue
        spk_diphone[(str(ph), str(prev))] = (float(r["mean"]), float(r["variance"]))

    spk_diphone_unstress_groups = defaultdict(list)  # (u_ph, u_prev) -> [(n,mu,var), ...]
    for _, r in dfd.iterrows():
        ph = r["phoneme"]
        prev = r["prev_phoneme"]
        if pd.isna(ph) or pd.isna(prev):
            continue
        ukey = (unstress(ph), unstress(prev))
        spk_diphone_unstress_groups[ukey].append((float(r["n"]), float(r["mean"]), float(r["variance"])))

    spk_diphone_unstress = {}
    for ukey, triples in spk_diphone_unstress_groups.items():
        pooled = pool_mu_var_from_triples(triples)  # (mu,var,N)
        if pooled is not None:
            spk_diphone_unstress[ukey] = pooled

    pooled_all = None
    dfa_all = df[df["stat_type"] == "all_phonemes"]
    if len(dfa_all) > 0:
        pooled_all = _pool_mu_var_from_group(dfa_all)

    pooled_phone = {}
    dfp_all = df[df["stat_type"] == "by_phoneme"].dropna(subset=["phoneme"])
    if len(dfp_all) > 0:
        for ph, g in dfp_all.groupby("phoneme"):
            pooled = _pool_mu_var_from_group(g)
            if pooled is not None:
                pooled_phone[str(ph)] = pooled

    pooled_diphone = {}
    dfd_all = df[df["stat_type"] == "by_phoneme_and_prev"].dropna(subset=["phoneme", "prev_phoneme"])
    if len(dfd_all) > 0:
        for (ph, prev), g in dfd_all.groupby(["phoneme", "prev_phoneme"]):
            pooled = _pool_mu_var_from_group(g)
            if pooled is not None:
                pooled_diphone[(str(ph), str(prev))] = pooled

    spk_diphone_rows = []  # each: (ph, prev, n, mean, var)
    for _, r in dfd.iterrows():
        ph = r["phoneme"]
        prev = r["prev_phoneme"]
        if pd.isna(ph) or pd.isna(prev):
            continue
        spk_diphone_rows.append((str(ph), str(prev), float(r["n"]), float(r["mean"]), float(r["variance"])))

    return (
        spk_all,
        spk_phone, spk_phone_unstress, spk_phone_rows,
        spk_diphone, spk_diphone_unstress, spk_diphone_rows,
        pooled_all, pooled_phone, pooled_diphone
    )


def pick_muvar_phoneme(
    ph,
    spk_phone,
    spk_phone_unstress,
    spk_phone_rows,
    phone_to_classes,
    class_to_phones,
    pooled_all,          # unused now (kept to minimize diffs)
    min_var_ref_phone    # unused now (kept to minimize diffs)
):
    # 1) exact stressed phone (ref speaker)
    muvar = spk_phone.get(ph)
    if muvar is not None:
        return (float(muvar[0]), float(muvar[1]))

    # 2) unstressed phone (ref speaker)
    ph_u = unstress(ph)
    muvarN = spk_phone_unstress.get(ph_u)
    if muvarN is not None and int(muvarN[2]) > 0:
        mu1, var1, _N1 = muvarN
        return (float(mu1), float(var1))

    # 3) fine class backoff
    fine_set = phones_in_same_class(ph_u, class_to_phones, phone_to_classes)
    assert len(fine_set) > 0, f"Meow: phoneme {ph} (unstressed={ph_u}) not found in any class."

    triples = []
    for ph_i, n_i, mu_i, var_i in spk_phone_rows:
        if unstress(ph_i) in fine_set:
            triples.append((n_i, mu_i, var_i))

    pooled = pool_mu_var_from_triples(triples)
    if pooled is not None and pooled[2] > 0:
        mu2, var2, _N2 = pooled
        return (float(mu2), float(var2))

    # 4) coarse class backoff: VOWEL / CONSONANT / SPECIAL.<s>
    q_classes = phone_to_classes.get(ph_u, set())
    coarse_set = set()
    if "COARSE.VOWEL" in q_classes:
        coarse_set |= class_to_phones.get("COARSE.VOWEL", set())
    if "COARSE.CONSONANT" in q_classes:
        coarse_set |= class_to_phones.get("COARSE.CONSONANT", set())
    if "SPECIAL.<s>" in q_classes:
        coarse_set |= class_to_phones.get("SPECIAL.<s>", set())

    assert len(coarse_set) > 0, f"Meow: no COARSE class for ph={ph} (unstressed={ph_u})."

    triples2 = []
    for ph_i, n_i, mu_i, var_i in spk_phone_rows:
        if unstress(ph_i) in coarse_set:
            triples2.append((n_i, mu_i, var_i))

    pooled2 = pool_mu_var_from_triples(triples2)
    assert pooled2 is not None and pooled2[2] > 0, (
        f"Meow: no COARSE class-phone matches for ph={ph} (unstressed={ph_u})."
    )

    mu3, var3, _N3 = pooled2
    return (float(mu3), float(var3))


def pick_muvar_diphone(
    ph, prev,
    spk_diphone,
    spk_diphone_unstress,
    spk_diphone_rows,
    phone_to_classes,
    class_to_phones,
    pooled_all,            # unused now (kept to minimize diffs)
    min_var_ref_diphone    # unused now (kept to minimize diffs)
):
    # 1) exact stressed diphone (ref speaker)
    muvar = spk_diphone.get((ph, prev))
    if muvar is not None:
        return (float(muvar[0]), float(muvar[1]))

    # 2) same diphone, stress-stripped (ref speaker)
    ukey = (unstress(ph), unstress(prev))
    muvarN = spk_diphone_unstress.get(ukey)
    if muvarN is not None and int(muvarN[2]) > 0:
        mu_u, var_u, _N = muvarN
        return (float(mu_u), float(var_u))

    ph_u = unstress(ph)
    prev_u = unstress(prev)

    ph_class_set = phones_in_same_class(ph_u, class_to_phones, phone_to_classes)
    prev_class_set = phones_in_same_class(prev_u, class_to_phones, phone_to_classes)

    assert len(ph_class_set) > 0, f"Meow: phoneme {ph} (unstressed={ph_u}) not found in any class."
    assert len(prev_class_set) > 0, f"Meow: prev phoneme {prev} (unstressed={prev_u}) not found in any class."

    # 3) same first phone (exact), prev in same fine class set
    triples = []
    for ph_i, prev_i, n_i, mu_i, var_i in spk_diphone_rows:
        if ph_i != ph:
            continue
        if unstress(prev_i) in prev_class_set:
            triples.append((n_i, mu_i, var_i))

    pooled = pool_mu_var_from_triples(triples)
    if pooled is not None and pooled[2] > 0:
        mu_c, var_c, _Nc = pooled
        return (float(mu_c), float(var_c))

    # 4) both phones in same fine class sets
    triples2 = []
    for ph_i, prev_i, n_i, mu_i, var_i in spk_diphone_rows:
        if unstress(ph_i) in ph_class_set and unstress(prev_i) in prev_class_set:
            triples2.append((n_i, mu_i, var_i))

    pooled2 = pool_mu_var_from_triples(triples2)
    if pooled2 is not None and pooled2[2] > 0:
        mu2, var2, _N2 = pooled2
        return (float(mu2), float(var2))

    # 4b) COARSE both-side backoff
    q_ph_classes = phone_to_classes.get(ph_u, set())
    q_prev_classes = phone_to_classes.get(prev_u, set())

    ph_coarse_set = set()
    if "COARSE.VOWEL" in q_ph_classes:
        ph_coarse_set |= class_to_phones.get("COARSE.VOWEL", set())
    if "COARSE.CONSONANT" in q_ph_classes:
        ph_coarse_set |= class_to_phones.get("COARSE.CONSONANT", set())
    if "SPECIAL.<s>" in q_ph_classes:
        ph_coarse_set |= class_to_phones.get("SPECIAL.<s>", set())

    prev_coarse_set = set()
    if "COARSE.VOWEL" in q_prev_classes:
        prev_coarse_set |= class_to_phones.get("COARSE.VOWEL", set())
    if "COARSE.CONSONANT" in q_prev_classes:
        prev_coarse_set |= class_to_phones.get("COARSE.CONSONANT", set())
    if "SPECIAL.<s>" in q_prev_classes:
        prev_coarse_set |= class_to_phones.get("SPECIAL.<s>", set())

    assert len(ph_coarse_set) > 0, f"Meow: no COARSE class for ph={ph} (unstressed={ph_u})."
    assert len(prev_coarse_set) > 0, f"Meow: no COARSE class for prev={prev} (unstressed={prev_u})."

    triples3 = []
    for ph_i, prev_i, n_i, mu_i, var_i in spk_diphone_rows:
        if unstress(ph_i) in ph_coarse_set and unstress(prev_i) in prev_coarse_set:
            triples3.append((n_i, mu_i, var_i))

    pooled3 = pool_mu_var_from_triples(triples3)
    assert pooled3 is not None and pooled3[2] > 0, (
        f"Meow: no COARSE class-diphone matches for (ph={ph}, prev={prev}); "
        f"ph_u={ph_u} prev_u={prev_u}"
    )

    mu3, var3, _N3 = pooled3
    return (float(mu3), float(var3))


def main(args):
    wavlm = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, device=device)

    class_to_phones, phone_to_classes = load_phoneme_classes(args.phoneme_classes_json)

    libri_root = Path(args.librispeech_root)
    src_dir = libri_root / str(args.src_speaker)
    tgt_dir = libri_root / str(args.tgt_speaker)

    src_wavs = sorted(src_dir.rglob("*.flac"))
    tgt_wavs = sorted(tgt_dir.rglob("*.flac"))

    with open(args.alignments_json, "r") as f:
        alignments = json.load(f)

    alpha = args.alpha
    lam = args.lam
    k = int(args.segment_splits)
    min_len = int(args.min_segment_length or 1)

    # --------- build target bag ----------
    bag = []
    for wav_path in tqdm(tgt_wavs, desc="Building target bag"):
        utt_id = wav_path.stem

        wav, sr = torchaudio.load(wav_path)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)

        with torch.no_grad():
            x, _ = wavlm.extract_features(wav.to(device), output_layer=6)
        x = x.squeeze(0)

        if args.target_use_alignment:
            if utt_id not in alignments:
                raise KeyError(f"missing alignment for target file stem={utt_id}")

            current_frame = 0
            alignment = alignments[utt_id]
            for phoneme_info in alignment:
                start = math.floor(phoneme_info["start"] * FPS)
                end = math.ceil(phoneme_info["end"] * FPS)

                if args.bag_silence and start > current_frame:
                    silence_frames = x[current_frame:start]
                    for chunk in split_into_equal_portions(silence_frames, k):
                        bag.append(chunk)

                if start < end:
                    frames = x[start:end]
                    for chunk in split_into_equal_portions(frames, k):
                        bag.append(chunk)

                current_frame = max(current_frame, end)

            if args.bag_silence and current_frame < x.shape[0]:
                for chunk in split_into_equal_portions(x[current_frame:], k):
                    bag.append(chunk)
        else:
            for chunk in split_into_equal_portions(x, k):
                if int(chunk.shape[0]) >= min_len:
                    bag.append(chunk)

        if args.max_segment_length and args.max_segment_length > 0:
            for seg in all_subsegments_leq(x, args.max_segment_length):
                if int(seg.shape[0]) >= min_len:
                    bag.append(seg)

    if len(bag) == 0:
        raise RuntimeError("Meow: target bag is empty.")

    # --------- precompute bag reps ----------
    bag_means = bag_first = bag_last = None
    if args.unit_distance_type in ("mean", "mean_first_last"):
        with torch.no_grad():
            bag_means = torch.stack([length_scaled_pool(seg, alpha) for seg in bag], dim=0).to(device)
    if args.unit_distance_type == "mean_first_last":
        with torch.no_grad():
            bag_first = torch.stack([seg[0] for seg in bag], dim=0).to(device)
            bag_last = torch.stack([seg[-1] for seg in bag], dim=0).to(device)

    def pick_from_bag(query_frames, use_first=True, use_last=True):
        if query_frames is None or len(query_frames) == 0:
            return None

        if args.unit_distance_type == "dtw":
            best = None
            best_dist = float("inf")
            for item in bag:
                d = dtw(query_frames, item, lam)
                if d < best_dist:
                    best_dist = d
                    best = item
            return best

        if args.unit_distance_type == "mean":
            q = length_scaled_pool(query_frames, alpha).to(device)
            dists = torch.sum((bag_means - q.unsqueeze(0)) ** 2, dim=1)
            return bag[int(torch.argmin(dists).item())]

        if args.unit_distance_type == "mean_first_last":
            w_mean, w_first, w_last = args.w_mean, args.w_first, args.w_last
            q_mean = length_scaled_pool(query_frames, alpha).to(device)
            dist = w_mean * torch.sum((bag_means - q_mean.unsqueeze(0)) ** 2, dim=1)

            if use_first and w_first != 0.0:
                q_first = query_frames[0].to(device)
                dist = dist + w_first * torch.sum((bag_first - q_first.unsqueeze(0)) ** 2, dim=1)

            if use_last and w_last != 0.0:
                q_last = query_frames[-1].to(device)
                dist = dist + w_last * torch.sum((bag_last - q_last.unsqueeze(0)) ** 2, dim=1)

            return bag[int(torch.argmin(dist).item())]

        raise ValueError(f"Unknown unit_distance_type: {args.unit_distance_type}")

    # --------- build converted segments (RECOMBINE matched lengths for likelihood) ----------
    conv_segments = []  # each: {label, prev_label, dur_secs}

    for wav_path in tqdm(src_wavs, desc="Converting + collecting durations"):
        utt_id = wav_path.stem
        if utt_id not in alignments:
            continue
        ali = alignments[utt_id]
        if not ali:
            continue

        wav, sr = torchaudio.load(wav_path)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)

        with torch.no_grad():
            x, _ = wavlm.extract_features(wav.to(device), output_layer=6)
        x = x.squeeze(0)

        current_frame = 0
        prev_label = "<s>"
        num_segments = len(ali)

        for seg_idx, ph in enumerate(ali):
            is_first = (seg_idx == 0)
            is_last = (seg_idx == num_segments - 1)

            phone_label = str(ph["phoneme"])
            start = math.floor(ph["start"] * FPS)
            end = math.ceil(ph["end"] * FPS)

            # inter-phone silence -> SIL
            if start > current_frame and args.bag_silence:
                sil = x[current_frame:start]
                sil_chunks = split_into_equal_portions(sil, k)

                total_Lt = 0
                any_match = False
                for chunk in sil_chunks:
                    if chunk.shape[0] == 0:
                        continue
                    matched = pick_from_bag(chunk, use_first=not is_first, use_last=not is_last)
                    if matched is None or matched.numel() == 0:
                        continue
                    total_Lt += int(matched.shape[0])
                    any_match = True

                if any_match:
                    lab = "SIL"
                    conv_segments.append(dict(label=lab, prev_label=prev_label, dur_secs=total_Lt / FPS))
                    prev_label = lab

            if start >= end:
                continue

            frames = x[start:end]
            current_frame = end

            chunks = split_into_equal_portions(frames, k)
            total_Lt = 0
            any_match = False
            for chunk in chunks:
                if chunk.shape[0] == 0:
                    continue
                matched = pick_from_bag(chunk, use_first=not is_first, use_last=not is_last)
                if matched is None or matched.numel() == 0:
                    continue
                total_Lt += int(matched.shape[0])
                any_match = True

            if any_match:
                lab = phone_label
                conv_segments.append(dict(label=lab, prev_label=prev_label, dur_secs=total_Lt / FPS))
                prev_label = lab

        # trailing silence -> SIL
        if args.bag_silence and current_frame < x.shape[0]:
            tail = x[current_frame:]
            tail_chunks = split_into_equal_portions(tail, k)

            total_Lt = 0
            any_match = False
            for chunk in tail_chunks:
                if chunk.shape[0] == 0:
                    continue
                matched = pick_from_bag(chunk, use_first=True, use_last=False)
                if matched is None or matched.numel() == 0:
                    continue
                total_Lt += int(matched.shape[0])
                any_match = True

            if any_match:
                lab = "SIL"
                conv_segments.append(dict(label=lab, prev_label=prev_label, dur_secs=total_Lt / FPS))
                prev_label = lab

    if len(conv_segments) == 0:
        raise RuntimeError("Meow: no converted segments produced.")

    (spk_all,
     spk_phone, spk_phone_unstress, spk_phone_rows,
     spk_diphone, spk_diphone_unstress, spk_diphone_rows,
     pooled_all, pooled_phone, pooled_diphone) = load_gt_models_with_pooled(args.gt_stats_csv, args.tgt_speaker)

    if pooled_all is None:
        raise RuntimeError("Meow: GT must contain stat_type=all_phonemes rows (with n/mean/variance).")

    # --------- compute LOG-likelihoods ----------
    seg_ll_phone = []
    per_phone_lls = defaultdict(list)

    seg_ll_diphone = []
    per_diphone_lls = defaultdict(list)

    for seg in conv_segments:
        xd = float(seg["dur_secs"])
        ph = str(seg["label"])
        prev = str(seg["prev_label"])

        mu, var = pick_muvar_phoneme(
            ph,
            spk_phone,
            spk_phone_unstress,
            spk_phone_rows,
            phone_to_classes,
            class_to_phones,
            pooled_all,
            0.0
        )
        p = normal_pdf(xd, mu, var)
        ll = math.log(max(p, LOG_EPS))
        seg_ll_phone.append(ll)
        per_phone_lls[ph].append(ll)

        mu2, var2 = pick_muvar_diphone(
            ph, prev,
            spk_diphone,
            spk_diphone_unstress,
            spk_diphone_rows,
            phone_to_classes,
            class_to_phones,
            pooled_all,
            0.0
        )
        p2 = normal_pdf(xd, mu2, var2)
        ll2 = math.log(max(p2, LOG_EPS))
        seg_ll_diphone.append(ll2)
        per_diphone_lls[(ph, prev)].append(ll2)

    avg_ll_all_phonemes = float(np.mean(seg_ll_phone))
    avg_ll_each_phone = {ph: float(np.mean(v)) for ph, v in per_phone_lls.items()}
    avg_of_phone_avg_lls = float(np.mean(list(avg_ll_each_phone.values()))) if avg_ll_each_phone else np.nan

    avg_ll_all_diphones = float(np.mean(seg_ll_diphone)) if seg_ll_diphone else np.nan
    avg_ll_each_diphone = {k: float(np.mean(v)) for k, v in per_diphone_lls.items()}
    avg_of_diphone_avg_lls = float(np.mean(list(avg_ll_each_diphone.values()))) if avg_ll_each_diphone else np.nan

    speaker_for_rows = str(args.tgt_speaker)
    rows = []

    rows.append(dict(
        speaker=speaker_for_rows,
        stat_type="loglik_all_phonemes",
        phoneme=None, phoneme_unstressed=None, prev_phoneme=None,
        n=len(seg_ll_phone),
        mean=avg_ll_all_phonemes,
        variance=np.nan,
    ))

    rows.append(dict(
        speaker=speaker_for_rows,
        stat_type="loglik_by_phoneme_avg_of_avgs",
        phoneme=None, phoneme_unstressed=None, prev_phoneme=None,
        n=len(avg_ll_each_phone),
        mean=avg_of_phone_avg_lls,
        variance=np.nan,
    ))

    for ph, m in sorted(avg_ll_each_phone.items()):
        rows.append(dict(
            speaker=speaker_for_rows,
            stat_type="loglik_by_phoneme",
            phoneme=ph, phoneme_unstressed=None, prev_phoneme=None,
            n=len(per_phone_lls[ph]),
            mean=m,
            variance=np.nan,
        ))

    rows.append(dict(
        speaker=speaker_for_rows,
        stat_type="loglik_all_diphones",
        phoneme=None, phoneme_unstressed=None, prev_phoneme=None,
        n=len(seg_ll_diphone),
        mean=avg_ll_all_diphones,
        variance=np.nan,
    ))

    rows.append(dict(
        speaker=speaker_for_rows,
        stat_type="loglik_by_diphone_avg_of_avgs",
        phoneme=None, phoneme_unstressed=None, prev_phoneme=None,
        n=len(avg_ll_each_diphone),
        mean=avg_of_diphone_avg_lls,
        variance=np.nan,
    ))

    for (ph, prev), m in sorted(avg_ll_each_diphone.items()):
        rows.append(dict(
            speaker=speaker_for_rows,
            stat_type="loglik_by_diphone",
            phoneme=ph, phoneme_unstressed=None, prev_phoneme=prev,
            n=len(per_diphone_lls[(ph, prev)]),
            mean=m,
            variance=np.nan,
        ))

    df = pd.DataFrame(rows, columns=[
        "speaker", "stat_type", "phoneme", "phoneme_unstressed", "prev_phoneme", "n", "mean", "variance"
    ])
    df.to_csv(args.out_csv, index=False)

    print(f"Meow: wrote {args.out_csv}")
    print(f"Meow: segments scored (phones)   n={len(seg_ll_phone)}  avg_loglik={avg_ll_all_phonemes:.6e}")
    print(f"Meow: segments scored (diphones) n={len(seg_ll_diphone)}  avg_loglik={avg_ll_all_diphones:.6e}")
    print(f"Meow: avg of per-phone avgs      = {avg_of_phone_avg_lls:.6e}")
    print(f"Meow: avg of per-diphone avgs    = {avg_of_diphone_avg_lls:.6e}")


if __name__ == "__main__":
    args = check_argv()
    main(args)