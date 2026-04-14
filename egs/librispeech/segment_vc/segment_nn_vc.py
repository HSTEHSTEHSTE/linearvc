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
    parser.add_argument(
        "--min-segment-length",
        type=int,
        default=1,
        help="Minimum segment length in feature frames. Segments shorter than this are skipped. (Does not apply to aligned segments)"
    )
    parser.add_argument(
        "--source-segment-length",
        type=int,
        default=0,
        help="If >0, ignore forced alignment for the source and instead use sliding windows of this many feature frames."
    )
    parser.add_argument(
        "--source-stride",
        type=int,
        default=0,
        help="Stride (in feature frames) between consecutive source windows when --source-segment-length > 0."
    )
    parser.add_argument(
        "--target_use_alignment",
        action="store_true",
        help="If set, build target bag from forced-aligned segments (and optional silence). If not set, do not use alignments for target bag."
    )
    parser.add_argument("--w_mean", type=float, default=5.0, help="Weight for mean/pooled distance")
    parser.add_argument("--w_first", type=float, default=1.0, help="Weight for first-frame distance")
    parser.add_argument("--w_last", type=float, default=1.0, help="Weight for last-frame distance")
    return parser.parse_args()

def overlap_add_average(chunks, starts, total_len, device=None):
    """
    chunks: list of [Li, D] tensors (matched segments)
    starts: list of start indices (in output frame coordinates) for each chunk
    total_len: total number of frames in final sequence (in source frame coordinates)
    Returns: [total_len, D] tensor where overlapping frames are averaged.
    """
    assert len(chunks) == len(starts)
    if len(chunks) == 0:
        return None

    D = chunks[0].shape[1]
    dev = device if device is not None else chunks[0].device

    acc = torch.zeros((total_len, D), device=dev)
    cnt = torch.zeros((total_len, 1), device=dev)

    for seg, s in zip(chunks, starts):
        if seg is None or seg.numel() == 0:
            continue
        L = seg.shape[0]
        e = min(total_len, s + L)
        L_eff = e - s
        if L_eff <= 0:
            continue
        acc[s:e] += seg[:L_eff]
        cnt[s:e] += 1.0

    # avoid divide-by-zero; any uncovered frames become 0
    return acc / torch.clamp(cnt, min=1.0)

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
    min_len = int(args.min_segment_length or 1)

    src_wav_path = Path('/home/hltcoe/xli/ARTS/anon_baseline/data/LibriSpeech/test-clean/61/70968/61-70968-0000.flac')
    tgt_speaker_path = Path('/home/hltcoe/xli/ARTS/anon_baseline/data/LibriSpeech/test-clean/121')
    # src_wav_path = Path('/home/hltcoe/xli/ARTS/anon_baseline/data/LibriSpeech/test-clean/121/121726/121-121726-0000.flac')
    # tgt_speaker_path = Path('/home/hltcoe/xli/ARTS/anon_baseline/data/LibriSpeech/test-clean/61')
    tgt_speaker_wavs = list(tgt_speaker_path.rglob("*.flac"))

    with open('/home/hltcoe/xli/ARTS/linearvc/exp/asr/LibriSpeech/alignments/alignments.json', 'r') as file:
        alignments = json.load(file)

    bag = []
    # ---- stats: lengths (in frames) for silence vs voiced ----
    silence_lens = []
    voiced_lens = []
    for tgt_speaker_wav in tqdm(tgt_speaker_wavs):
        wav, sr = torchaudio.load(tgt_speaker_wav)
        with torch.no_grad():
            x, _ = wavlm.extract_features(wav.to(device), output_layer=6)
        x = x.squeeze(0)  # [T, D]

        if args.target_use_alignment:
            # add alignment-derived chunks (voiced + optional silence)
            current_frame = 0
            alignment = alignments[tgt_speaker_wav.stem]
            for phoneme_info in alignment:
                start = math.floor(phoneme_info['start'] * 50)
                end = math.ceil(phoneme_info['end'] * 50)

                if start > current_frame:
                    silence_frames = x[current_frame:start]
                    for chunk in split_into_equal_portions(silence_frames, k):
                        silence_lens.append(int(chunk.shape[0]))
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

        # add all subsegments up to max segment length (feature frames)
        if args.max_segment_length and args.max_segment_length > 0:
            for seg in all_subsegments_leq(x, args.max_segment_length):
                if chunk.shape[0] >= min_len:
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
    src_alignment = alignments[src_wav_path.stem]

    # ---- bin source chunk length -> matched target chunk length counts ----
    src_len_to_tgt_len_counts = defaultdict(Counter)

    seg_len = int(args.source_segment_length or 0)
    stride = int(args.source_stride or 0)

    if seg_len <= 0:
        # ============== FALLBACK: original alignment-based source segmentation ==============
        new_seq_list = []
        current_frame = 0
        num_segments = len(src_alignment)

        for seg_idx, phoneme_info in enumerate(tqdm(src_alignment)):
            is_first_seg = (seg_idx == 0)
            is_last_seg = (seg_idx == num_segments - 1)

            start = math.floor(phoneme_info["start"] * 50)
            end = math.ceil(phoneme_info["end"] * 50)

            # optional silence region between phones
            if start > current_frame:
                silence_frames = x[current_frame:start]
                if args.bag_silence:
                    for chunk in split_into_equal_portions(silence_frames, k):
                        matched = pick_from_bag(chunk, use_first=not is_first_seg, use_last=not is_last_seg)
                        new_seq_list.append(matched)
                else:
                    new_seq_list.append(silence_frames)

            if start >= end:
                continue

            frames = x[start:end]
            for chunk in split_into_equal_portions(frames, k):
                matched = pick_from_bag(chunk, use_first=not is_first_seg, use_last=not is_last_seg)

                src_len = int(chunk.shape[0])
                tgt_len = int(matched.shape[0]) if matched is not None else 0
                src_len_to_tgt_len_counts[src_len][tgt_len] += 1

                new_seq_list.append(matched)

            current_frame = end

        # trailing silence after last phone
        if current_frame < x.shape[0]:
            tail = x[current_frame:]
            if args.bag_silence:
                for chunk in split_into_equal_portions(tail, k):
                    new_seq_list.append(pick_from_bag(chunk, use_first=True, use_last=False))
            else:
                new_seq_list.append(tail)

        new_seq = torch.cat(new_seq_list, dim=0)

    else:
        # ============== Sliding-window source segmentation + overlap-add stitching ==============
        if stride <= 0:
            raise ValueError("Meow: set --source-stride > 0 when --source-segment-length > 0.")

        T = x.shape[0]
        if T == 0:
            raise ValueError("Meow: source features have length 0 frames.")

        if T <= seg_len:
            starts = [0]
        else:
            starts = list(range(0, T - seg_len + 1, stride))
            if len(starts) == 0:
                starts = [0]
            if starts[-1] + seg_len < T:
                starts.append(T - seg_len)

        matched_chunks = []
        for i, s in enumerate(tqdm(starts)):
            e = min(T, s + seg_len)
            query = x[s:e]

            is_first = (i == 0)
            is_last = (i == len(starts) - 1)

            matched = pick_from_bag(query, use_first=not is_first, use_last=not is_last)

            src_len = int(query.shape[0])
            tgt_len = int(matched.shape[0]) if matched is not None else 0
            src_len_to_tgt_len_counts[src_len][tgt_len] += 1

            matched_chunks.append(matched)

        if len(matched_chunks) == 0:
            raise ValueError("Meow: no matched chunks produced; check seg_len/stride.")

        # place chunks based on matched lengths, preserving overlap=seg_len-stride
        matched_starts = []
        overlap = max(0, seg_len - stride)

        prev_start = 0
        prev_len = int(matched_chunks[0].shape[0]) if (matched_chunks[0] is not None and matched_chunks[0].numel() > 0) else 0
        matched_starts.append(0)

        for i in range(1, len(matched_chunks)):
            prev_end = prev_start + prev_len
            next_start = max(0, prev_end - overlap)

            matched_starts.append(next_start)

            prev_start = next_start
            seg = matched_chunks[i]
            prev_len = int(seg.shape[0]) if (seg is not None and seg.numel() > 0) else 0

        total_len = prev_start + prev_len
        new_seq = overlap_add_average(matched_chunks, matched_starts, total_len=total_len, device=device)
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