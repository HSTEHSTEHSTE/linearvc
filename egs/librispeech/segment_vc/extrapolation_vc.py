import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

device = "cuda"
EPS_VAR = 1e-8
STRESS_RE = re.compile(r"(\d)$")

VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
    "IH", "IY", "OW", "OY", "UH", "UW"
}
PAUSE_LABELS = {"SIL", "SP", "SPN", "<sil>", "PAU"}  # adjust if needed, meow


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


def is_spn(ph: str) -> bool:
    ph = canon(ph)
    return ph.lower() == "spn" or unstress(ph).lower() == "spn"


def is_start(ph: str) -> bool:
    return canon(ph) == "<s>"


def is_special(ph: str) -> bool:
    return is_spn(ph) or is_start(ph)


def stress_bucket(ph: str) -> str:
    ph = canon(ph)
    if is_special(ph):
        return "NO_STRESS"
    d = stress_digit(ph)
    if d is None:
        return "NO_STRESS"
    return "V0" if d == 0 else "VS"


def stress_compatible(query_ph: str, cand_ph: str) -> bool:
    query_ph = canon(query_ph)
    cand_ph = canon(cand_ph)

    if is_special(query_ph):
        return cand_ph == query_ph

    qd = stress_digit(query_ph)
    if qd is None:
        return True

    return stress_bucket(query_ph) == stress_bucket(cand_ph)


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
        raise TypeError(f"Meow: unexpected node type in phoneme_classes.json at {path_parts}: {type(node)}")

    walk(obj, [])

    class_to_phones["COARSE.SPECIAL"] = {"<s>", "spn"}
    phone_to_classes["<s>"].add("COARSE.SPECIAL")
    phone_to_classes["spn"].add("COARSE.SPECIAL")

    vowel_set = set()
    for cname, phones in class_to_phones.items():
        if str(cname).startswith("vowels."):
            vowel_set |= set(phones)

    consonant_set = set()
    for cname, phones in class_to_phones.items():
        cname = str(cname)
        if cname.startswith("vowels.") or cname.startswith("COARSE."):
            continue
        consonant_set |= set(phones)

    class_to_phones["COARSE.VOWEL"] = vowel_set
    class_to_phones["COARSE.CONSONANT"] = consonant_set

    for p in vowel_set:
        phone_to_classes[p].add("COARSE.VOWEL")
    for p in consonant_set:
        phone_to_classes[p].add("COARSE.CONSONANT")

    return class_to_phones, phone_to_classes


def fine_class_set(base_phone: str, class_to_phones, phone_to_classes):
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


def load_gt_models_ref_only_norm(gt_csv: str, speaker_id: str):
    df = pd.read_csv(gt_csv, low_memory=False)
    df["speaker"] = df["speaker"].astype(str)
    spk_df = df[df["speaker"] == str(speaker_id)]

    df_ar = spk_df[spk_df["stat_type"] == "articulation_rate_gaussian"]
    if len(df_ar) == 0:
        raise KeyError(f"Meow: no articulation_rate_gaussian for speaker={speaker_id} in {gt_csv}")
    r0 = df_ar.iloc[0]
    ar_mu = float(r0["mean"])
    ar_var = float(r0["variance"])
    ar_n = int(r0["n"])

    spk_diphone = {}
    dfd = spk_df[spk_df["stat_type"] == "by_phoneme_and_prev_norm_by_avg_syll_len"]
    spk_diphone_rows = []
    for _, r in dfd.iterrows():
        ph, prev = r["phoneme"], r["prev_phoneme"]
        if pd.isna(ph) or pd.isna(prev):
            continue
        ph = canon(ph)
        prev = canon(prev)
        spk_diphone[(ph, prev)] = (float(r["mean"]), float(r["variance"]))
        spk_diphone_rows.append((ph, prev, float(r["n"]), float(r["mean"]), float(r["variance"])))

    return (ar_mu, ar_var, ar_n), spk_diphone, spk_diphone_rows


def pick_muvar_diphone(
    ph, prev,
    spk_diphone,
    spk_diphone_rows,
    phone_to_classes,
    class_to_phones,
):
    ph = canon(ph)
    prev = canon(prev)

    muvar = spk_diphone.get((ph, prev))
    if muvar is not None:
        return (float(muvar[0]), float(muvar[1]), "EXACT")

    ph_u = unstress(ph)
    prev_u = unstress(prev)

    ph_fine = {ph} if is_special(ph) else fine_class_set(ph_u, class_to_phones, phone_to_classes)
    prev_fine = {prev} if is_special(prev) else fine_class_set(prev_u, class_to_phones, phone_to_classes)

    if not is_special(ph) and len(ph_fine) == 0:
        raise RuntimeError(f"Meow: no fine class for ph={ph} base={ph_u}")
    if not is_special(prev) and len(prev_fine) == 0:
        raise RuntimeError(f"Meow: no fine class for prev={prev} base={prev_u}")

    ph_coarse = coarse_tag(ph, phone_to_classes)
    prev_coarse = coarse_tag(prev, phone_to_classes)
    if ph_coarse is None:
        raise RuntimeError(f"Meow: no coarse class for ph={ph} base={ph_u}")
    if prev_coarse is None:
        raise RuntimeError(f"Meow: no coarse class for prev={prev} base={prev_u}")

    # 2) current fine, prev exact
    triples2 = []
    for ph_i, prev_i, n_i, mu_i, var_i in spk_diphone_rows:
        if prev_i != prev:
            continue
        if (unstress(ph_i) in ph_fine) and stress_compatible(ph, ph_i) and stress_compatible(prev, prev_i):
            triples2.append((n_i, mu_i, var_i))
    pooled2 = pool_mu_var_from_triples(triples2)
    if pooled2 is not None and pooled2[2] > 0:
        return (float(pooled2[0]), float(pooled2[1]), "CURR_FINE__PREV_EXACT")

    # 3) current fine, prev fine
    triples3 = []
    for ph_i, prev_i, n_i, mu_i, var_i in spk_diphone_rows:
        if (unstress(ph_i) in ph_fine) and (unstress(prev_i) in prev_fine):
            if stress_compatible(ph, ph_i) and stress_compatible(prev, prev_i):
                triples3.append((n_i, mu_i, var_i))
    pooled3 = pool_mu_var_from_triples(triples3)
    if pooled3 is not None and pooled3[2] > 0:
        return (float(pooled3[0]), float(pooled3[1]), "CURR_FINE__PREV_FINE")

    # 4) current coarse, prev fine
    triples4 = []
    for ph_i, prev_i, n_i, mu_i, var_i in spk_diphone_rows:
        if coarse_tag(ph_i, phone_to_classes) != ph_coarse:
            continue
        if unstress(prev_i) not in prev_fine:
            continue
        if stress_compatible(ph, ph_i) and stress_compatible(prev, prev_i):
            triples4.append((n_i, mu_i, var_i))
    pooled4 = pool_mu_var_from_triples(triples4)
    if pooled4 is not None and pooled4[2] > 0:
        return (float(pooled4[0]), float(pooled4[1]), "CURR_COARSE__PREV_FINE")

    # 5) current coarse, prev coarse
    triples5 = []
    for ph_i, prev_i, n_i, mu_i, var_i in spk_diphone_rows:
        if coarse_tag(ph_i, phone_to_classes) != ph_coarse:
            continue
        if coarse_tag(prev_i, phone_to_classes) != prev_coarse:
            continue
        if stress_compatible(ph, ph_i) and stress_compatible(prev, prev_i):
            triples5.append((n_i, mu_i, var_i))
    pooled5 = pool_mu_var_from_triples(triples5)
    if pooled5 is not None and pooled5[2] > 0:
        return (float(pooled5[0]), float(pooled5[1]), "CURR_COARSE__PREV_COARSE")

    raise KeyError(f"Meow: no diphone stats found after backoff for (ph={ph}, prev={prev}).")


def load_audio_mono(path: Path, sr: int = 16000):
    wav, in_sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if in_sr != sr:
        wav = torchaudio.functional.resample(wav, in_sr, sr)
    return wav, sr


def get_models():
    wavlm = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, device=device)
    wavlm.eval()
    hifigan, _ = torch.hub.load(
        "bshall/knn-vc", "hifigan_wavlm", trust_repo=True, device=device, prematched=True
    )
    hifigan.eval()
    return wavlm, hifigan


@torch.no_grad()
def wavlm_feats(wavlm, wav_1ch: torch.Tensor, output_layer: int):
    x, _ = wavlm.extract_features(wav_1ch.to(device), output_layer=output_layer)
    return x.squeeze(0)  # [T, D]


def build_target_frame_pool(wavlm, tgt_files, output_layer: int, sr: int):
    all_frames = []
    for f in tqdm(tgt_files, desc="Meow: building target frame pool"):
        wav, _ = load_audio_mono(f, sr=sr)
        x = wavlm_feats(wavlm, wav, output_layer=output_layer)  # [T, D]
        if x.numel() > 0:
            all_frames.append(x)
    if len(all_frames) == 0:
        raise ValueError("Meow: target speaker pool is empty.")
    return torch.cat(all_frames, dim=0).contiguous()  # [N, D]


def nn_map_frames(src_frames: torch.Tensor, tgt_pool: torch.Tensor, chunk: int = 200000):
    src_n = F.normalize(src_frames, p=2, dim=1)
    mapped = []

    for i in tqdm(range(src_n.shape[0]), desc="Meow: per-frame NN"):
        q = src_n[i: i + 1]  # [1, D]
        best_sim = None
        best_vec = None

        for s in range(0, tgt_pool.shape[0], chunk):
            tp = tgt_pool[s: s + chunk]
            tp_n = F.normalize(tp, p=2, dim=1)
            sims = (q @ tp_n.T).squeeze(0)
            j = int(torch.argmax(sims).item())
            sim = sims[j].item()
            if (best_sim is None) or (sim > best_sim):
                best_sim = sim
                best_vec = tp[j]

        mapped.append(best_vec.unsqueeze(0))

    return torch.cat(mapped, dim=0)  # [T, D]


def alignment_segments_with_silence_and_gap(alignment, T: int, fps: float = 50.0, sil_gap_sec: float = 0.02):
    """
    Returns list of dict segments:
      {start, end, start_sec, end_sec, phoneme, prev}
    """
    segs = []
    prev_phone = "<s>"
    sil_gap_frames = int(math.ceil(float(sil_gap_sec) * float(fps)))

    prev_end = None  # frames

    for phseg in alignment:
        ph = canon(phseg.get("phoneme", "spn"))
        s = int(math.floor(float(phseg["start"]) * fps))
        e = int(math.ceil(float(phseg["end"]) * fps))

        s = max(0, min(s, T))
        e = max(0, min(e, T))

        if prev_end is None:
            if s > sil_gap_frames:
                segs.append(dict(
                    start=0, end=s,
                    start_sec=0.0, end_sec=float(s) / fps,
                    phoneme="spn", prev=prev_phone
                ))
                prev_phone = "spn"
        else:
            gap = s - prev_end
            if gap > sil_gap_frames:
                segs.append(dict(
                    start=prev_end, end=s,
                    start_sec=float(prev_end) / fps, end_sec=float(s) / fps,
                    phoneme="spn", prev=prev_phone
                ))
                prev_phone = "spn"

        if e > s:
            segs.append(dict(
                start=s, end=e,
                start_sec=float(s) / fps, end_sec=float(e) / fps,
                phoneme=ph, prev=prev_phone
            ))
            prev_phone = ph
            prev_end = e
        else:
            prev_end = max(prev_end, e) if prev_end is not None else e

    if prev_end is None:
        prev_end = 0
    tail = T - prev_end
    if tail > sil_gap_frames:
        segs.append(dict(
            start=prev_end, end=T,
            start_sec=float(prev_end) / fps, end_sec=float(T) / fps,
            phoneme="spn", prev=prev_phone
        ))

    return [z for z in segs if z["end"] > z["start"]]


def resample_segment_linear(seg: torch.Tensor, out_len: int):
    L, D = seg.shape
    out_len = int(out_len)
    if out_len <= 0:
        raise ValueError("Meow: out_len must be > 0.")
    if L == 0:
        raise ValueError("Meow: empty segment.")
    if L == 1:
        return seg.expand(out_len, D)
    if out_len == L:
        return seg
    x = seg.transpose(0, 1).unsqueeze(0)
    y = F.interpolate(x, size=out_len, mode="linear", align_corners=True)
    return y.squeeze(0).transpose(0, 1).contiguous()


def sample_gaussian(mu: float, var: float, rng: np.random.Generator):
    sigma = math.sqrt(max(float(var), EPS_VAR))
    return float(rng.normal(loc=float(mu), scale=sigma))


def sample_duration_norm(mu: float, var: float, rng: np.random.Generator, min_norm: float = 0.05):
    x = sample_gaussian(mu, var, rng=rng)
    return max(float(x), float(min_norm))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--src_wav", type=str,
                    default="/home/hltcoe/xli/ARTS/anon_baseline/data/LibriSpeech/test-clean/61/70968/61-70968-0000.flac")
    ap.add_argument("--tgt_speaker_dir", type=str,
                    default="/home/hltcoe/xli/ARTS/anon_baseline/data/LibriSpeech/test-clean/121")
    ap.add_argument("--alignments_json", type=str,
                    default="/home/hltcoe/xli/ARTS/linearvc/exp/asr/LibriSpeech/alignments/alignments.json")

    ap.add_argument("--gt_stats_csv", type=str,
                    default="duration/gt/norm_stats.csv")
    ap.add_argument("--phoneme_classes_json", type=str, default="duration/phoneme_classes.json")
    ap.add_argument("--tgt_speaker", type=str, default="121")

    ap.add_argument("--out_wav", type=str, default="output_diphone_sampled.wav")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--output_layer", type=int, default=6)
    ap.add_argument("--tgt_glob", type=str, default="*.flac")
    ap.add_argument("--chunk", type=int, default=200000)

    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--sil_gap", type=float, default=0.02)
    ap.add_argument("--min_out_frames", type=int, default=1)
    ap.add_argument("--min_norm", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--segments_log", type=str, default="segments_log.txt")

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    class_to_phones, phone_to_classes = load_phoneme_classes(args.phoneme_classes_json)

    with open(args.alignments_json, "r") as f:
        alignments = json.load(f)

    src_wav_path = Path(args.src_wav)
    if src_wav_path.stem not in alignments:
        raise KeyError(f"Meow: no alignment entry for key '{src_wav_path.stem}' in {args.alignments_json}")

    (ar_stats, spk_diphone, spk_diphone_rows) = load_gt_models_ref_only_norm(args.gt_stats_csv, args.tgt_speaker)
    ar_mu, ar_var, ar_n = ar_stats

    ar_syll_per_sec = sample_gaussian(ar_mu, ar_var, rng=rng)
    ar_syll_per_sec = max(ar_syll_per_sec, 1e-6)
    avg_syll_len_sec = 1.0 / ar_syll_per_sec

    print(f"Meow: target speaker={args.tgt_speaker} articulation_rate sample = {ar_syll_per_sec:.6f} syll/sec "
          f"(mu={ar_mu:.6f}, var={ar_var:.6g}, n={ar_n})")
    print(f"Meow: implied avg_syll_len = {avg_syll_len_sec:.6f} sec/syll")

    wavlm, hifigan = get_models()

    src_wav, _ = load_audio_mono(src_wav_path, sr=args.sr)
    with torch.no_grad():
        src_x = wavlm_feats(wavlm, src_wav, output_layer=args.output_layer)
    T, D = src_x.shape
    if T == 0:
        raise ValueError("Meow: source has 0 WavLM frames.")

    tgt_dir = Path(args.tgt_speaker_dir)
    tgt_files = sorted(tgt_dir.rglob(args.tgt_glob))
    tgt_pool = build_target_frame_pool(wavlm, tgt_files, output_layer=args.output_layer, sr=args.sr)

    mapped_frames = nn_map_frames(src_x, tgt_pool, chunk=args.chunk)

    ali = alignments[src_wav_path.stem]
    segs = alignment_segments_with_silence_and_gap(ali, T=T, fps=args.fps, sil_gap_sec=args.sil_gap)

    seg_log_path = Path(args.segments_log) if args.segments_log else None
    seg_log_f = None
    if seg_log_path is not None:
        seg_log_path.parent.mkdir(parents=True, exist_ok=True)
        seg_log_f = open(seg_log_path, "w", encoding="utf-8")
        seg_log_f.write(
            "Meow: segment log\n"
            f"Meow: src={src_wav_path}\n"
            f"Meow: tgt_speaker={args.tgt_speaker}\n"
            f"Meow: ar_syll_per_sec={ar_syll_per_sec:.6f}  avg_syll_len_sec={avg_syll_len_sec:.6f}\n"
            f"Meow: fps={args.fps}\n"
            f"Meow: sil_gap={args.sil_gap}\n"
            "Meow: backoff levels: EXACT | CURR_FINE__PREV_EXACT | CURR_FINE__PREV_FINE | CURR_COARSE__PREV_FINE | CURR_COARSE__PREV_COARSE\n"
            "----\n"
        )

    out_list = []
    total_in = 0
    total_out = 0

    for idx, seg in enumerate(segs):
        s, e = int(seg["start"]), int(seg["end"])
        s_sec, e_sec = float(seg["start_sec"]), float(seg["end_sec"])
        ph, prev = canon(seg["phoneme"]), canon(seg["prev"])
        chunk = mapped_frames[s:e]
        L = int(chunk.shape[0])
        if L <= 0:
            continue

        mu_norm, var_norm, backoff = pick_muvar_diphone(
            ph, prev,
            spk_diphone,
            spk_diphone_rows,
            phone_to_classes,
            class_to_phones,
        )
        # dur_norm = sample_duration_norm(mu_norm, var_norm, rng=rng, min_norm=args.min_norm)
        dur_norm = max(float(mu_norm), float(args.min_norm))  # meow: use mean, no sampling
        dur_s = dur_norm * avg_syll_len_sec

        out_len = int(math.ceil(dur_s * float(args.fps)))
        out_len = max(int(args.min_out_frames), out_len)

        chunk2 = resample_segment_linear(chunk, out_len=out_len)
        out_list.append(chunk2)

        total_in += L
        total_out += out_len

        in_s = L / float(args.fps)
        out_s = out_len / float(args.fps)

        line = (
            f"seg#{idx:04d} t=[{s_sec:8.3f},{e_sec:8.3f}] "
            f"frames=[{s:6d}:{e:6d}] ph={ph:>6s} prev={prev:>6s} "
            f"in={L:4d}f ({in_s:.3f}s) -> out={out_len:4d}f ({out_s:.3f}s) "
            f"backoff={backoff:<24s} "
            f"dur_norm={dur_norm:.4f} (mu={mu_norm:.4f}, var={var_norm:.4g})"
        )

        print(f"Meow: {line}")
        if seg_log_f is not None:
            seg_log_f.write(line + "\n")

    if seg_log_f is not None:
        seg_log_f.write("----\n")
        seg_log_f.write(
            f"totals in={total_in} frames ({total_in/float(args.fps):.3f}s) "
            f"out={total_out} frames ({total_out/float(args.fps):.3f}s)\n"
        )
        seg_log_f.close()

    if len(out_list) == 0:
        raise RuntimeError("Meow: produced no segments after alignment.")

    new_seq = torch.cat(out_list, dim=0)

    with torch.no_grad():
        wav_hat = hifigan(new_seq.unsqueeze(0)).squeeze(0).detach().cpu()

    if wav_hat.dim() == 1:
        wav_hat = wav_hat.unsqueeze(0)

    out_path = Path(args.out_wav)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_path), wav_hat, args.sr)

    print(f"Meow: totals in={total_in} frames ({total_in/float(args.fps):.3f}s) "
          f"out={total_out} frames ({total_out/float(args.fps):.3f}s)")
    print(f"Meow: wrote {out_path}  (samples={wav_hat.shape[-1]}, sr={args.sr})")
    if seg_log_path is not None:
        print(f"Meow: wrote segment log to {seg_log_path}")


if __name__ == "__main__":
    main()