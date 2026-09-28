import math
import json
from pathlib import Path
from collections import Counter

import torch
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm
import re

# ---------------- config ----------------
ROOT = Path("/home/hltcoe/xli/ARTS/anon_baseline/data/LibriSpeech/test-clean")
ALIGN_PATH = Path("/home/hltcoe/xli/ARTS/linearvc/exp/asr/LibriSpeech/alignments/alignments.json")

SRC_SPK = "61"
TGT_SPK = "121"
LAYER = 6
FPS = 50
BATCH = 2048
device = "cuda" if torch.cuda.is_available() else "cpu"

_STRESS_RE = re.compile(r"^([A-Z]+)([0-2])$")


def rms_normalize_mono(x: torch.Tensor, target_dbfs=-20.0, eps=1e-8):
    # x: (1, T)
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)  # (1,1)
    target_rms = 10.0 ** (target_dbfs / 20.0)
    gain = target_rms / (rms + eps)
    return x * gain


def collapse_stress(ph: str) -> str:
    m = _STRESS_RE.match(ph)
    return m.group(1) if m else ph


def load_alignments():
    with open(ALIGN_PATH, "r") as f:
        return json.load(f)


def get_segment_bounds_frames(alignment, fps=50):
    """Return list of (segidx, start_frame, end_frame, phoneme_collapsed)."""
    segs = []
    for k, seg in enumerate(alignment):
        s = int(math.floor(seg["start"] * fps))
        e = int(math.ceil(seg["end"] * fps))
        if s >= e:
            continue
        segs.append((k, s, e, collapse_stress(seg["phoneme"])))
    return segs


def extract_wavlm_feats(wavlm, wav_path: Path, layer=LAYER):
    wav, sr = torchaudio.load(wav_path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000
    wav = rms_normalize_mono(wav, target_dbfs=-10.0)
    with torch.no_grad():
        x, _ = wavlm.extract_features(wav.to(device), output_layer=layer)
    return x.squeeze(0)  # [T, D]


def batched_nn_cosine(src_feats, tgt_feats, batch=BATCH):
    src = F.normalize(src_feats, dim=1)
    tgt = F.normalize(tgt_feats, dim=1)

    out = []
    for i in tqdm(range(0, src.shape[0], batch), desc="Per-frame NN", leave=False):
        s = src[i:i + batch]
        sims = s @ tgt.T
        out.append(torch.argmax(sims, dim=1).detach().cpu())
    return torch.cat(out, dim=0)  # [N] indices into tgt


def mean_pool(seg_feats):
    return seg_feats.mean(dim=0)


def build_target_segment_bank(wavlm, alignments, tgt_speaker_dir: Path):
    """
    Returns:
      tgt_seg_means: [K, D] normalized, GPU
      tgt_all_feats: [M, D] GPU
      tgt_frame_to_segk: [M] CPU (segk or -1)
      tgt_frame_to_pos_in_seg: [M] CPU (0..len-1) or -1
      tgt_seg_lengths: [K] CPU (len in frames)
    """
    wav_paths = sorted(tgt_speaker_dir.rglob("*.flac"))

    seg_means = []
    seg_lengths = []

    all_feats = []
    frame_to_segk = []
    frame_to_pos = []

    segk_counter = 0

    for wav_path in tqdm(wav_paths, desc=f"Target feats+segments {tgt_speaker_dir.name}"):
        utt = wav_path.stem
        if utt not in alignments:
            continue
        alignment = alignments[utt]

        x = extract_wavlm_feats(wavlm, wav_path)  # [T, D]
        T = x.shape[0]

        f_segk = torch.full((T,), -1, dtype=torch.long)
        f_pos = torch.full((T,), -1, dtype=torch.long)

        segs = get_segment_bounds_frames(alignment, fps=FPS)
        for (segidx, s, e, ph) in segs:
            s = max(s, 0)
            e = min(e, T)
            if s >= e:
                continue

            seg_feat = x[s:e]
            mu = mean_pool(seg_feat)
            seg_means.append(mu)
            seg_lengths.append(e - s)

            # map frames to this segment index and position within segment
            f_segk[s:e] = segk_counter
            f_pos[s:e] = torch.arange(0, e - s, dtype=torch.long)

            segk_counter += 1

        all_feats.append(x)
        frame_to_segk.append(f_segk)
        frame_to_pos.append(f_pos)

    tgt_all_feats = torch.cat(all_feats, dim=0).to(device)               # [M, D]
    tgt_frame_to_segk = torch.cat(frame_to_segk, dim=0).cpu()            # [M]
    tgt_frame_to_pos_in_seg = torch.cat(frame_to_pos, dim=0).cpu()       # [M]

    tgt_seg_means = torch.stack(seg_means, dim=0).to(device)             # [K, D]
    tgt_seg_means = F.normalize(tgt_seg_means, dim=1)

    tgt_seg_lengths = torch.tensor(seg_lengths, dtype=torch.long)        # [K]

    return (
        tgt_seg_means,
        tgt_all_feats,
        tgt_frame_to_segk,
        tgt_frame_to_pos_in_seg,
        tgt_seg_lengths,
    )


def main():
    wavlm = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, device=device)
    alignments = load_alignments()

    src_dir = ROOT / SRC_SPK
    tgt_dir = ROOT / TGT_SPK

    # ----- target bank -----
    (
        tgt_seg_means,
        tgt_all_feats,
        tgt_frame_to_segk,
        tgt_frame_to_pos,
        tgt_seg_lengths,
    ) = build_target_segment_bank(wavlm, alignments, tgt_dir)

    print(f"Target segments: {tgt_seg_means.shape[0]}, target frames: {tgt_all_feats.shape[0]}")

    # histogram over thresholds: 10,20,...,100
    thresholds = list(range(10, 101, 10))
    num_meeting = {t: 0 for t in thresholds}

    total_src_segments = 0

    # total aligned frames (pooled)
    total_frames_considered = 0
    total_frames_aligned = 0

    # ----- iterate source utterances/segments -----
    src_wavs = sorted(src_dir.rglob("*.flac"))

    for wav_path in tqdm(src_wavs, desc=f"Source segments {SRC_SPK}"):
        utt = wav_path.stem
        if utt not in alignments:
            continue
        alignment = alignments[utt]

        x = extract_wavlm_feats(wavlm, wav_path)  # [T, D]
        T = x.shape[0]

        # per-frame NN into ALL target frames
        nn_idx = batched_nn_cosine(x, tgt_all_feats, batch=BATCH)  # [T] CPU

        segs = get_segment_bounds_frames(alignment, fps=FPS)
        for (segidx, s, e, ph) in segs:
            s = max(s, 0)
            e = min(e, T)
            if s >= e:
                continue

            L = e - s
            total_src_segments += 1

            # best target segment by mean pooled cosine
            src_mu = F.normalize(mean_pool(x[s:e]).to(device), dim=0)  # [D]
            sims = tgt_seg_means @ src_mu                              # [K]
            best_segk = int(torch.argmax(sims).item())

            # For each source frame within this segment,
            # count aligned if NN target frame is inside best_segk segment.
            tgt_frames = nn_idx[s:e]                          # [L] indices into target frames
            mapped_segk = tgt_frame_to_segk[tgt_frames]       # [L] segk or -1

            aligned = (mapped_segk == best_segk) & (mapped_segk >= 0)

            aligned_count = int(aligned.sum().item())
            frac_aligned = aligned_count / L

            # update threshold histogram
            pct = frac_aligned * 100.0
            for t in thresholds:
                if pct >= t:
                    num_meeting[t] += 1

            total_frames_considered += L
            total_frames_aligned += aligned_count

    print(f"\n=== Result ({SRC_SPK} -> {TGT_SPK}) ===")
    print(f"Total source segments: {total_src_segments}")
    for t in thresholds:
        pct_segments = (100.0 * num_meeting[t] / total_src_segments) if total_src_segments > 0 else 0.0
        print(f"Segments with >= {t}% aligned frames: {num_meeting[t]} ({pct_segments:.2f}%)")

    if total_frames_considered > 0:
        print(
            f"\nOverall aligned-frame rate (pooled over segments): "
            f"{total_frames_aligned}/{total_frames_considered} = "
            f"{100.0 * total_frames_aligned / total_frames_considered:.2f}%"
        )


if __name__ == "__main__":
    main()