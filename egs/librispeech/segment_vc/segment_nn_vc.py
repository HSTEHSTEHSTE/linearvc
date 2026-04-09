import argparse
from collections import defaultdict, Counter
import math
from pathlib import Path
import pickle
from tqdm import tqdm
import numpy as np
import torch, torchaudio
import torch.nn.functional as F
import json

device = "cuda"

def check_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unit_distance_type",
        type=str,
        choices=["mean", "dtw", "mean_first_last"],
        help="mean, dtw, mean_first_last",
        default="mean_first_last"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help="mean pooled length scaling",
        default=0.,
    )
    parser.add_argument(
        "--lam",
        type=float,
        help="DTW length penalty",
        default=0.1,
    )
    parser.add_argument(
        "--bag_silence",
        action='store_true',
        help="put silence into the bag of hypotheses",
    )
    parser.add_argument(
        "--segment_splits",
        type=int,
        default=1,
        help="Split each segment into this many equal portions (default: 1).",
    )
    parser.add_argument(
        "--max-segment-length",
        type=int,
        default=0,
        help="If >0, also add ALL contiguous segments from each target-speaker audio whose length is <= this many frames."
    )
    parser.add_argument("--w_mean", type=float, default=1.0, help="Weight for mean/pooled distance")
    parser.add_argument("--w_first", type=float, default=1.0, help="Weight for first-frame distance")
    parser.add_argument("--w_last", type=float, default=1.0, help="Weight for last-frame distance")
    return parser.parse_args()

def length_scaled_pool(X, alpha):
    n = X.shape[0]
    return X.mean(axis=0) * (n ** alpha)

def dtw(X, Y, lam=0.0):
    n, m = len(X), len(Y)

    X_norm = F.normalize(X, p=2, dim=1)
    Y_norm = F.normalize(Y, p=2, dim=1)
    cost_matrix = 1 - X_norm @ Y_norm.T  # cosine distance

    D = torch.full((n + 1, m + 1), float('inf'), device=X.device)
    D[0, 0] = 0.0
    lam = torch.tensor(lam, device=X.device)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = cost_matrix[i-1, j-1]
            D[i, j] = cost + min(
                D[i-1, j-1],        # match
                D[i-1, j] + lam,    # skip in Y
                D[i, j-1] + lam     # skip in X
            )

    return D[n, m] / 1000

def split_into_equal_portions(frames, k: int):
    """Split [T, D] into k contiguous chunks (nearly equal length)."""
    if frames is None or len(frames) == 0:
        return []
    k = max(1, int(k))
    if k == 1:
        return [frames]
    # torch.tensor_split yields nearly equal chunks; some may be empty if k > T
    chunks = torch.tensor_split(frames, k, dim=0)
    return [c for c in chunks if c.numel() > 0]

def _print_len_stats(name, lens):
    if len(lens) == 0:
        print(f"{name}: no segments")
        return
    mn = min(lens)
    mx = max(lens)
    avg = sum(lens) / len(lens)
    print(f"{name}: min={mn} max={mx} avg={avg:.2f} (n={len(lens)})")

def all_subsegments_leq(x: torch.Tensor, max_len: int):
    """
    x: [T, D]
    yields: [L, D] for all i, L where 1<=L<=max_len and i+L<=T
    """
    T = x.shape[0]
    max_len = min(max_len, T)
    for i in range(T):
        max_L = min(max_len, T - i)
        for L in range(1, max_L + 1):
            yield x[i:i+L]

def main(args):
    wavlm = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, device=device)
    hifigan, _ = torch.hub.load("bshall/knn-vc", "hifigan_wavlm", trust_repo=True, device=device, prematched=True)
    alpha = args.alpha
    lam = args.lam
    k = args.segment_splits

    # src_wav_path = Path('/home/hltcoe/xli/ARTS/anon_baseline/data/LibriSpeech/test-clean/61/70968/61-70968-0000.flac')
    # tgt_speaker_path = Path('/home/hltcoe/xli/ARTS/anon_baseline/data/LibriSpeech/test-clean/121')
    src_wav_path = Path('/home/hltcoe/xli/ARTS/anon_baseline/data/LibriSpeech/test-clean/121/121726/121-121726-0000.flac')
    tgt_speaker_path = Path('/home/hltcoe/xli/ARTS/anon_baseline/data/LibriSpeech/test-clean/61')
    tgt_speaker_wavs = list(tgt_speaker_path.rglob("*.flac"))

    with open('/home/hltcoe/xli/ARTS/linearvc/exp/asr/LibriSpeech/alignments/alignments.json', 'r') as file:
        alignments = json.load(file)

    bag = []
    # ---- stats: lengths (in frames) for silence vs voiced ----
    silence_lens = []
    voiced_lens = []
    for tgt_speaker_wav in tqdm(tgt_speaker_wavs):
        alignment = alignments[tgt_speaker_wav.stem]
        wav, sr = torchaudio.load(tgt_speaker_wav)
        with torch.no_grad():
            x, _ = wavlm.extract_features(wav.to(device), output_layer=6)
        x = x.squeeze(0)  # [T, D]

        # 1) existing: add alignment-derived chunks (voiced + optional silence)
        current_frame = 0
        for phoneme_info in alignment:
            start = math.floor(phoneme_info['start'] * 50)
            end = math.ceil(phoneme_info['end'] * 50)

            if start > current_frame:
                silence_frames = x[current_frame:start]
                for chunk in split_into_equal_portions(silence_frames, k):
                    silence_lens.append(int(chunk.shape[0]))
                    if args.bag_silence:
                        bag.append(chunk)

            if start >= end:
                continue

            frames = x[start:end]
            current_frame = end
            for chunk in split_into_equal_portions(frames, k):
                voiced_lens.append(int(chunk.shape[0]))
                bag.append(chunk)

        # trailing silence after last phone (optional; your code currently doesn't add it)
        if args.bag_silence and current_frame < x.shape[0]:
            for chunk in split_into_equal_portions(x[current_frame:], k):
                silence_lens.append(int(chunk.shape[0]))
                bag.append(chunk)

        # 2) NEW: also add all subsegments up to max segment length (feature frames)
        if args.max_segment_length and args.max_segment_length > 0:
            for seg in all_subsegments_leq(x, args.max_segment_length):
                bag.append(seg)
    
    bag_means = None
    bag_first = None
    bag_last = None

    if args.unit_distance_type in ("mean", "mean_first_last"):
        with torch.no_grad():
            bag_means = torch.stack([length_scaled_pool(seg, alpha) for seg in bag], dim=0).to(device)  # [B, D]

    if args.unit_distance_type == "mean_first_last":
        with torch.no_grad():
            bag_first = torch.stack([seg[0] for seg in bag], dim=0).to(device)  # [B, D]
            bag_last  = torch.stack([seg[-1] for seg in bag], dim=0).to(device) # [B, D]
    
    def pick_from_bag(query_frames, use_first=True, use_last=True):
        """Return best-matching segment from bag according to unit_distance_type."""
        if len(query_frames) == 0:
            return query_frames  # nothing to match

        if args.unit_distance_type == "dtw":
            min_dist = float("inf")
            target_seq = None
            for item in bag:
                dist = dtw(query_frames, item, lam)
                if dist < min_dist:
                    min_dist = dist
                    target_seq = item
            return target_seq

        elif args.unit_distance_type == "mean":
            q = length_scaled_pool(query_frames, alpha).to(device)              # [D]
            dists = torch.sum((bag_means - q.unsqueeze(0)) ** 2, dim=1)         # [B]
            idx = torch.argmin(dists).item()
            return bag[idx]

        elif args.unit_distance_type == "mean_first_last":
            w_mean, w_first, w_last = args.w_mean, args.w_first, args.w_last

            q_mean = length_scaled_pool(query_frames, alpha).to(device)         # [D]
            dist = w_mean * torch.sum((bag_means - q_mean.unsqueeze(0)) ** 2, dim=1)  # [B]

            if use_first and w_first != 0.0:
                q_first = query_frames[0].to(device)                            # [D]
                dist = dist + w_first * torch.sum((bag_first - q_first.unsqueeze(0)) ** 2, dim=1)

            if use_last and w_last != 0.0:
                q_last = query_frames[-1].to(device)                            # [D]
                dist = dist + w_last * torch.sum((bag_last - q_last.unsqueeze(0)) ** 2, dim=1)

            idx = torch.argmin(dist).item()
            return bag[idx]

        else:
            raise ValueError(f"Unknown unit_distance_type: {args.unit_distance_type}")

    wav, sr = torchaudio.load(src_wav_path)
    with torch.no_grad():
        x, _ = wavlm.extract_features(wav.to(device), output_layer=6)
    x = x.squeeze(0)

    alignment = alignments[src_wav_path.stem]
    # ---- bin source voiced chunk length -> matched target chunk length counts ----
    src_len_to_tgt_len_counts = defaultdict(Counter)
    new_seq = []
    current_frame = 0

    num_segments = len(alignment)
    for seg_idx, phoneme_info in enumerate(tqdm(alignment)):
        is_first_seg = (seg_idx == 0)
        is_last_seg = (seg_idx == num_segments - 1)
        start = math.floor(phoneme_info['start'] * 50)
        end = math.ceil(phoneme_info['end'] * 50)

        if start > current_frame:
            silence_frames = x[current_frame:start]
            if args.bag_silence:
                for chunk in split_into_equal_portions(silence_frames, k):
                    new_seq.append(pick_from_bag(chunk, use_first=not is_first_seg, use_last=not is_last_seg))
            else:
                new_seq.append(silence_frames)

        if start >= end:
            continue

        frames = x[start:end]
        for chunk in split_into_equal_portions(frames, k):
            src_len = int(chunk.shape[0])
            matched = pick_from_bag(chunk, use_first=not is_first_seg, use_last=not is_last_seg)
            tgt_len = int(matched.shape[0]) if matched is not None else 0

            src_len_to_tgt_len_counts[src_len][tgt_len] += 1
            new_seq.append(matched)

        current_frame = end

    new_seq = torch.cat(new_seq, axis=0)
    with torch.no_grad():
        wav_hat = hifigan(new_seq.unsqueeze(0)).squeeze(0).detach().cpu()
    torchaudio.save('output.wav', wav_hat, 16000)

    _print_len_stats("Voiced segments (frames)", voiced_lens)
    _print_len_stats("Silence segments (frames)", silence_lens)

    # ---- dump bins to TSV: rows=src_len, cols=tgt_len ----
    out_path = Path("length_bins.csv")

    src_lens = sorted(src_len_to_tgt_len_counts.keys())
    all_tgt_lens = sorted({t for c in src_len_to_tgt_len_counts.values() for t in c.keys()})

    with out_path.open("w") as f:
        # header: split=<k> then target lengths
        f.write("split=" + str(k))
        for t in all_tgt_lens:
            f.write("\t" + str(t))
        f.write("\n")

        # rows
        for s in src_lens:
            f.write(str(s))
            row = src_len_to_tgt_len_counts[s]
            for t in all_tgt_lens:
                val = row.get(t, 0)
                f.write("\t" + ("" if val == 0 else str(val)))
            f.write("\n")

if __name__ == "__main__":
    args = check_argv()
    main(args)