from pathlib import Path
import json
import itertools
import re
import math
from collections import defaultdict, Counter
import csv
import torch, torchaudio
import torch.nn.functional as F
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

ROOT = Path("/home/hltcoe/xli/ARTS/anon_baseline/data/LibriSpeech/test-clean")
ALIGN_PATH = Path("/home/hltcoe/xli/ARTS/linearvc/exp/asr/LibriSpeech/alignments/alignments.json")

def load_alignments():
    with open(ALIGN_PATH, "r") as f:
        return json.load(f)

_STRESS_RE = re.compile(r"^([A-Z]+)([0-2])$")

def collapse_stress(ph: str) -> str:
    m = _STRESS_RE.match(ph)
    return m.group(1) if m else ph

def segs_to_frame_labels(alignment, num_frames, fps=50, collapse=True):
    # include silence frames in phoneme analyses
    labels = ["silence"] * num_frames
    for seg in alignment:
        ph = seg["phoneme"]
        if collapse:
            ph = collapse_stress(ph)
        s = int(math.floor(seg["start"] * fps))
        e = int(math.ceil(seg["end"] * fps))
        s = max(s, 0)
        e = min(e, num_frames)
        for t in range(s, e):
            labels[t] = ph
    return labels

def segs_to_frame_segment_ids(utt_id, alignment, num_frames, fps=50):
    # used for segment-diversity; "NONE" = silence/uncovered
    seg_ids = ["NONE"] * num_frames
    for k, seg in enumerate(alignment):
        s = int(math.floor(seg["start"] * fps))
        e = int(math.ceil(seg["end"] * fps))
        s = max(s, 0)
        e = min(e, num_frames)
        sid = f"{utt_id}:{k}"
        for t in range(s, e):
            seg_ids[t] = sid
    return seg_ids

def extract_speaker_frames_with_segments(wavlm, alignments, speaker_dir, layer=6, fps=50):
    wav_paths = sorted(speaker_dir.rglob("*.flac"))

    all_feats = []
    utt_ids, frame_idxs, phonemes, seg_ids = [], [], [], []
    seginfo = {}

    for wav_path in tqdm(wav_paths, desc=f"Extract {speaker_dir.name}"):
        utt = wav_path.stem
        if utt not in alignments:
            continue
        alignment = alignments[utt]

        wav, sr = torchaudio.load(wav_path)
        with torch.no_grad():
            x, _ = wavlm.extract_features(wav.to(device), output_layer=layer)
        x = x.squeeze(0)  # [T, D]
        T = x.shape[0]

        frame_ph = segs_to_frame_labels(alignment, T, fps=fps, collapse=True)
        frame_seg = segs_to_frame_segment_ids(utt, alignment, T, fps=fps)

        # segment metadata (optional, kept for compatibility)
        for k, seg in enumerate(alignment):
            s = int(math.floor(seg["start"] * fps))
            e = int(math.ceil(seg["end"] * fps))
            if s >= e:
                continue
            sid = f"{utt}:{k}"
            if sid not in seginfo:
                seginfo[sid] = {
                    "utt": utt,
                    "segidx": k,
                    "phoneme": collapse_stress(seg["phoneme"]),
                    "start_frame": max(s, 0),
                    "end_frame": min(e, T),
                }

        all_feats.append(x)
        utt_ids.extend([utt] * T)
        frame_idxs.extend(list(range(T)))
        phonemes.extend(frame_ph)
        seg_ids.extend(frame_seg)

    feats = torch.cat(all_feats, dim=0) if all_feats else torch.empty(0, 1024, device=device)
    meta = {"utt_id": utt_ids, "frame_idx": frame_idxs, "phoneme": phonemes, "seg_id": seg_ids}
    return feats, meta, seginfo

def batched_nn_cosine(src_feats, tgt_feats, batch=2048):
    """
    Nearest neighbor per source frame in target frames by cosine similarity.
    Returns: 1D CPU LongTensor of indices into tgt_feats.
    """
    src = F.normalize(src_feats, dim=1)
    tgt = F.normalize(tgt_feats, dim=1)

    nn_idx = []
    for i in tqdm(range(0, src.shape[0], batch), desc="NN search", leave=False):
        s = src[i:i+batch]          # [b, D]
        sims = s @ tgt.T            # [b, M]
        idx = torch.argmax(sims, dim=1)
        nn_idx.append(idx.detach().cpu())
    return torch.cat(nn_idx, dim=0)

# ---------- aggregation helpers ----------

def update_confusion_agg(conf_agg, src_labels, tgt_labels, nn_idx):
    # conf_agg[src_label][tgt_label] += count
    for i in range(len(nn_idx)):
        s = src_labels[i]
        t = tgt_labels[int(nn_idx[i])]
        conf_agg[s][t] += 1

def update_adjacency_agg(adj_agg, src_meta, tgt_meta, nn_idx):
    # adj_agg: dict with keys total_pairs, preserved
    N = len(nn_idx)
    for i in range(N - 1):
        if src_meta["utt_id"][i] != src_meta["utt_id"][i + 1]:
            continue
        if src_meta["frame_idx"][i + 1] != src_meta["frame_idx"][i] + 1:
            continue

        adj_agg["total_pairs"] += 1

        j0 = int(nn_idx[i])
        j1 = int(nn_idx[i + 1])
        if tgt_meta["utt_id"][j0] == tgt_meta["utt_id"][j1] and \
           tgt_meta["frame_idx"][j1] == tgt_meta["frame_idx"][j0] + 1:
            adj_agg["preserved"] += 1

def update_segment_div_bins_agg(hist, src_meta, tgt_meta, nn_idx, max_bin=100):
    """
    Segment-diversity bins pooled across all pairs.
    Excludes silence: skip frames with seg_id=="NONE" on either side.
    hist bin key: "1".."99","100+"
    """
    hits = defaultdict(set)  # src_seg_id -> set(tgt_seg_id)

    N = len(nn_idx)
    for i in range(N):
        s_seg = src_meta["seg_id"][i]
        if s_seg == "NONE":
            continue
        j = int(nn_idx[i])
        t_seg = tgt_meta["seg_id"][j]
        if t_seg == "NONE":
            continue
        hits[s_seg].add(t_seg)

    for _, tset in hits.items():
        k = len(tset)
        b = f"{max_bin}+" if k >= max_bin else str(k)
        hist[b] += 1

def write_confusion_csv_from_agg(conf_agg, out_csv, normalize_rows=False):
    labels = sorted(set(conf_agg.keys()) | {t for c in conf_agg.values() for t in c.keys()})
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src\\tgt"] + labels)
        for s in labels:
            row = [conf_agg.get(s, Counter()).get(t, 0) for t in labels]
            if normalize_rows:
                denom = sum(row)
                row = [(c / denom) if denom > 0 else 0.0 for c in row]
            w.writerow([s] + row)

def write_segdiv_bins_csv_from_agg(hist, out_csv, max_bin=100):
    total_segments = sum(hist.values())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bin", "num_source_segments", "pct_source_segments"])
        w.writeheader()
        for k in range(1, max_bin):
            b = str(k)
            n = hist.get(b, 0)
            w.writerow({
                "bin": b,
                "num_source_segments": n,
                "pct_source_segments": (n / total_segments) if total_segments > 0 else 0.0
            })
        b = f"{max_bin}+"
        n = hist.get(b, 0)
        w.writerow({
            "bin": b,
            "num_source_segments": n,
            "pct_source_segments": (n / total_segments) if total_segments > 0 else 0.0
        })

# ---------- main pooling logic ----------

def main():
    wavlm = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, device=device)
    alignments = load_alignments()

    # list all speaker dirs in test-clean
    speaker_dirs = sorted([p for p in ROOT.iterdir() if p.is_dir() and p.name.isdigit()])
    speaker_ids = [p.name for p in speaker_dirs]
    print(f"Found {len(speaker_ids)} speakers in test-clean")

    # extract once per speaker and keep in memory (fastest, but uses lots of GPU/CPU RAM)
    speaker_cache = {}
    for spk in tqdm(speaker_ids, desc="Extract all speakers"):
        feats, meta, _ = extract_speaker_frames_with_segments(wavlm, alignments, ROOT / spk, layer=6)
        speaker_cache[spk] = (feats, meta)

    # global accumulators (pooled across all pairs, both directions)
    conf_agg = defaultdict(Counter)          # includes "silence" frames
    segdiv_hist = Counter()                 # excludes silence (NONE) as requested
    adj_agg = {"total_pairs": 0, "preserved": 0}

    # iterate unordered pairs, but update stats for both directions
    pairs = list(itertools.combinations(speaker_ids, 2))
    for a, b in tqdm(pairs, desc="All speaker pairs"):
        featsA, metaA = speaker_cache[a]
        featsB, metaB = speaker_cache[b]

        if featsA.numel() == 0 or featsB.numel() == 0:
            continue

        # A -> B
        nnA2B = batched_nn_cosine(featsA, featsB, batch=2048)
        update_confusion_agg(conf_agg, metaA["phoneme"], metaB["phoneme"], nnA2B)
        update_segment_div_bins_agg(segdiv_hist, metaA, metaB, nnA2B, max_bin=100)
        update_adjacency_agg(adj_agg, metaA, metaB, nnA2B)

        # B -> A
        nnB2A = batched_nn_cosine(featsB, featsA, batch=2048)
        update_confusion_agg(conf_agg, metaB["phoneme"], metaA["phoneme"], nnB2A)
        update_segment_div_bins_agg(segdiv_hist, metaB, metaA, nnB2A, max_bin=100)
        update_adjacency_agg(adj_agg, metaB, metaA, nnB2A)

    # write pooled outputs
    write_confusion_csv_from_agg(conf_agg, "confusion_allpairs_counts.csv", normalize_rows=False)
    write_confusion_csv_from_agg(conf_agg, "confusion_allpairs_rowprob.csv", normalize_rows=True)

    write_segdiv_bins_csv_from_agg(segdiv_hist, "segment_div_bins_allpairs.csv", max_bin=100)

    # pooled adjacency number
    total_pairs = adj_agg["total_pairs"]
    preserved = adj_agg["preserved"]
    rate = preserved / total_pairs if total_pairs > 0 else 0.0
    print(f"\nAdjacency preservation (pooled, both directions): {preserved}/{total_pairs} = {rate*100:.2f}%")

    print("Wrote: confusion_allpairs_counts.csv, confusion_allpairs_rowprob.csv, segment_div_bins_allpairs.csv")

if __name__ == "__main__":
    main()