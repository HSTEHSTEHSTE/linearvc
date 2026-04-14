import argparse
import random
import math
from pathlib import Path
from tqdm import tqdm
import torch, torchaudio
import torch.nn.functional as F
import json
from cuvs.neighbors import brute_force

device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def make_bag_cat(bag_means, bag_first, bag_last, args, use_first=True, use_last=True):
    parts = []
    if args.w_mean != 0.0:
        parts.append((args.w_mean ** 0.5) * bag_means.float())
    if use_first and args.w_first != 0.0:
        parts.append((args.w_first ** 0.5) * bag_first.float())
    if use_last and args.w_last != 0.0:
        parts.append((args.w_last ** 0.5) * bag_last.float())
    return torch.cat(parts, dim=1).contiguous()  # [N, Dcat]

@torch.no_grad()
def build_exact_bf_index(bag_cat: torch.Tensor):
    X_cp = bag_cat
    index = brute_force.build(X_cp, metric="l2")  # pass dataset here
    return index

@torch.no_grad()
def exact_1nn(index, q_cat: torch.Tensor) -> int:
    q_cp = q_cat.unsqueeze(0)  # [1, D]
    # cuVS brute_force.search returns (distances, indices)
    d_cp, i_cp = brute_force.search(index, q_cp, k=1)
    i_t = torch.as_tensor(i_cp)
    return int(i_t.item())


def check_argv():
    p = argparse.ArgumentParser()

    p.add_argument("--src_dir", type=Path, required=True, help="Root with speaker subdirs containing audio (recursive).")
    p.add_argument("--out_dir", type=Path, required=True, help="Output root; writes under out_dir/<spk_src>/...")

    p.add_argument("--seed", type=int, default=42, help="Seed for derangement mapping.")

    p.add_argument("--alignments_json", type=Path, required=True, help="Forced alignments JSON keyed by file stem.")

    # matching
    p.add_argument(
        "--unit_distance_type",
        type=str,
        choices=["mean", "dtw", "mean_first_last"],
        default="mean_first_last"
    )
    p.add_argument("--alpha", type=float, default=0.0)
    p.add_argument("--lam", type=float, default=0.1)

    # bag building
    p.add_argument("--bag_silence", action="store_true")
    p.add_argument("--segment_splits", type=int, default=1)
    p.add_argument("--max-segment-length", type=int, default=0)
    p.add_argument("--min-segment-length", type=int, default=1)
    p.add_argument("--target_use_alignment", action="store_true")

    # source segmentation
    p.add_argument("--source-segment-length", type=int, default=0)
    p.add_argument("--source-stride", type=int, default=0)

    # mean_first_last weights
    p.add_argument("--w_mean", type=float, default=1.0)
    p.add_argument("--w_first", type=float, default=1.0)
    p.add_argument("--w_last", type=float, default=1.0)

    return p.parse_args()


def get_spk_mapping(spks, seed):
    random.seed(seed)
    if len(spks) < 2:
        raise ValueError("Not enough speakers")

    while True:
        shuffled = spks[:]
        random.shuffle(shuffled)
        if all(a != b for a, b in zip(spks, shuffled)):
            return dict(zip(spks, shuffled))


def list_audio_files(root: Path):
    exts = (".wav", ".flac", ".mp3")
    files = []
    for e in exts:
        files.extend(root.rglob(f"*{e}"))
    return sorted(files)


def list_speakers(src_dir: Path):
    spks = []
    for p in sorted(src_dir.iterdir()):
        if p.is_dir():
            spks.append(p.name)
    return spks


def overlap_add_average(chunks, starts, total_len, device=None):
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
    return acc / torch.clamp(cnt, min=1.0)


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
            D[i, j] = cost + min(
                D[i - 1, j - 1],
                D[i - 1, j] + lam,
                D[i, j - 1] + lam,
            )
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
    T = x.shape[0]
    max_len = min(max_len, T)
    for i in range(T):
        max_L = min(max_len, T - i)
        for L in range(1, max_L + 1):
            yield x[i : i + L]


def build_target_bag(args, wavlm, alignments, tgt_spk_dir: Path):
    k = args.segment_splits
    min_len = int(args.min_segment_length or 1)

    tgt_wavs = list_audio_files(tgt_spk_dir)
    if len(tgt_wavs) == 0:
        raise ValueError(f"no target audio under {tgt_spk_dir}")

    bag = []
    for tgt_wav in tgt_wavs:
        wav, _ = torchaudio.load(tgt_wav)
        with torch.no_grad():
            x, _ = wavlm.extract_features(wav.to(device), output_layer=6)
        x = x.squeeze(0).detach().cpu()  # [T, D]

        if args.target_use_alignment:
            if tgt_wav.stem not in alignments:
                raise KeyError(f"missing alignment for target file stem={tgt_wav.stem}")

            current_frame = 0
            alignment = alignments[tgt_wav.stem]
            for phoneme_info in alignment:
                start = math.floor(phoneme_info["start"] * 50)
                end = math.ceil(phoneme_info["end"] * 50)

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

        if args.max_segment_length and args.max_segment_length > 0:
            for seg in all_subsegments_leq(x, args.max_segment_length):
                if seg.shape[0] >= min_len:
                    bag.append(seg)

    if len(bag) == 0:
        raise ValueError("built empty target bag; check options.")
    return bag


def make_picker(args, bag):
    alpha = args.alpha
    lam = args.lam

    bag_means = bag_first = bag_last = None
    if args.unit_distance_type in ("mean", "mean_first_last"):
        with torch.no_grad():
            bag_means = torch.stack([length_scaled_pool(seg, alpha) for seg in bag], dim=0).to(device)
    if args.unit_distance_type == "mean_first_last":
        with torch.no_grad():
            bag_first = torch.stack([seg[0] for seg in bag], dim=0).to(device)
            bag_last = torch.stack([seg[-1] for seg in bag], dim=0).to(device)

    bf_full = bf_mf = bf_ml = bf_m = None
    if args.unit_distance_type == "mean_first_last":
        # exact brute-force indices for each boundary condition (matches old logic)
        bag_m = ((args.w_mean ** 0.5) * bag_means.float()).contiguous()
        bf_m = build_exact_bf_index(bag_m)

        # bag_mf = torch.cat([
        #     (args.w_mean ** 0.5) * bag_means.float(),
        #     (args.w_first ** 0.5) * bag_first.float(),
        # ], dim=1).contiguous()
        # bf_mf = build_exact_bf_index(bag_mf)

        # bag_ml = torch.cat([
        #     (args.w_mean ** 0.5) * bag_means.float(),
        #     (args.w_last ** 0.5) * bag_last.float(),
        # ], dim=1).contiguous()
        # bf_ml = build_exact_bf_index(bag_ml)

        bag_full = torch.cat([
            (args.w_mean ** 0.5) * bag_means.float(),
            (args.w_first ** 0.5) * bag_first.float(),
            (args.w_last ** 0.5) * bag_last.float(),
        ], dim=1).contiguous()
        bag_full = bag_full.half()
        bf_full = build_exact_bf_index(bag_full)

    def pick_from_bag(query_frames, use_first=True, use_last=True):
        if len(query_frames) == 0:
            return query_frames

        if args.unit_distance_type == "dtw":
            min_dist = float("inf")
            target_seq = None
            for item in bag:
                dist = dtw(query_frames, item, lam)
                if dist < min_dist:
                    min_dist = dist
                    target_seq = item
            return target_seq

        if args.unit_distance_type == "mean":
            q = length_scaled_pool(query_frames, alpha).to(device)
            dists = torch.sum((bag_means - q.unsqueeze(0)) ** 2, dim=1)
            return bag[int(torch.argmin(dists).item())]

        if args.unit_distance_type == "mean_first_last":
            q_mean = length_scaled_pool(query_frames, alpha).to(device).float()
            q_first = query_frames[0].to(device).float()
            q_last  = query_frames[-1].to(device).float()

            if use_first and use_last:
                q_cat = torch.cat([
                    (args.w_mean ** 0.5) * q_mean,
                    (args.w_first ** 0.5) * q_first,
                    (args.w_last ** 0.5) * q_last,
                ], dim=0).contiguous()
                q_cat = q_cat.half()
                idx = exact_1nn(bf_full, q_cat)

            elif use_first and (not use_last):
                q_cat = torch.cat([
                    (args.w_mean ** 0.5) * q_mean,
                    (args.w_first ** 0.5) * q_first,
                ], dim=0).contiguous()
                idx = exact_1nn(bf_mf, q_cat)

            elif (not use_first) and use_last:
                q_cat = torch.cat([
                    (args.w_mean ** 0.5) * q_mean,
                    (args.w_last ** 0.5) * q_last,
                ], dim=0).contiguous()
                idx = exact_1nn(bf_ml, q_cat)

            else:
                q_cat = ((args.w_mean ** 0.5) * q_mean).contiguous()
                idx = exact_1nn(bf_m, q_cat)

            return bag[idx]

        raise ValueError(f"unknown unit_distance_type: {args.unit_distance_type}")

    return pick_from_bag


def convert_one(args, wavlm, hifigan, alignments, pick_from_bag, src_wav_path: Path):
    wav, _ = torchaudio.load(src_wav_path)
    with torch.no_grad():
        x, _ = wavlm.extract_features(wav.to(device), output_layer=6)
    x = x.squeeze(0)

    seg_len = int(args.source_segment_length or 0)
    stride = int(args.source_stride or 0)
    k = args.segment_splits

    if seg_len <= 0:
        if src_wav_path.stem not in alignments:
            raise KeyError(f"missing alignment for source file stem={src_wav_path.stem}")
        src_alignment = alignments[src_wav_path.stem]

        new_seq_list = []
        current_frame = 0
        num_segments = len(src_alignment)

        for seg_idx, phoneme_info in enumerate(src_alignment):
            is_first_seg = (seg_idx == 0)
            is_last_seg = (seg_idx == num_segments - 1)

            start = math.floor(phoneme_info["start"] * 50)
            end = math.ceil(phoneme_info["end"] * 50)

            if start > current_frame:
                silence_frames = x[current_frame:start]
                if args.bag_silence:
                    for chunk in split_into_equal_portions(silence_frames, k):
                        matched = pick_from_bag(chunk, use_first=True, use_last=True)
                        new_seq_list.append(matched)
                else:
                    new_seq_list.append(silence_frames)

            if start < end:
                frames = x[start:end]
                for chunk in split_into_equal_portions(frames, k):
                    matched = pick_from_bag(chunk, use_first=True, use_last=True)
                    new_seq_list.append(matched)

            current_frame = max(current_frame, end)

        if current_frame < x.shape[0]:
            tail = x[current_frame:]
            if args.bag_silence:
                for chunk in split_into_equal_portions(tail, k):
                    new_seq_list.append(pick_from_bag(chunk, use_first=True, use_last=True))
            else:
                new_seq_list.append(tail)

        new_seq = torch.cat(new_seq_list, dim=0)

    else:
        if stride <= 0:
            raise ValueError("set --source-stride > 0 when --source-segment-length > 0.")
        T = x.shape[0]
        if T == 0:
            raise ValueError("source features have length 0 frames.")

        if T <= seg_len:
            starts = [0]
        else:
            starts = list(range(0, T - seg_len + 1, stride))
            if starts[-1] + seg_len < T:
                starts.append(T - seg_len)

        matched_chunks = []
        for i, s in enumerate(starts):
            e = min(T, s + seg_len)
            query = x[s:e]
            is_first = (i == 0)
            is_last = (i == len(starts) - 1)
            matched = pick_from_bag(query, use_first=True, use_last=True)
            matched_chunks.append(matched)

        overlap = max(0, seg_len - stride)

        matched_starts = [0]
        prev_start = 0
        prev_len = int(matched_chunks[0].shape[0]) if matched_chunks[0] is not None else 0
        for i in range(1, len(matched_chunks)):
            prev_end = prev_start + prev_len
            next_start = max(0, prev_end - overlap)
            matched_starts.append(next_start)
            prev_start = next_start
            prev_len = int(matched_chunks[i].shape[0]) if matched_chunks[i] is not None else 0

        total_len = prev_start + prev_len
        new_seq = overlap_add_average(matched_chunks, matched_starts, total_len=total_len, device=device)

    with torch.no_grad():
        wav_hat = hifigan(new_seq.unsqueeze(0).to(device)).squeeze(0).detach().cpu()
    return wav_hat


def main(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.alignments_json, "r") as f:
        alignments = json.load(f)

    wavlm = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, device=device)
    hifigan, _ = torch.hub.load("bshall/knn-vc", "hifigan_wavlm", trust_repo=True, device=device, prematched=True)

    spks = list_speakers(args.src_dir)
    spk_map = get_spk_mapping(spks, args.seed)

    for spk_src in tqdm(spks, desc="speakers"):
        spk_tgt = spk_map[spk_src]

        src_spk_dir = args.src_dir / spk_src
        tgt_spk_dir = args.src_dir / spk_tgt

        bag = build_target_bag(args, wavlm, alignments, tgt_spk_dir=tgt_spk_dir)
        pick_from_bag = make_picker(args, bag)

        src_files = list_audio_files(src_spk_dir)
        for src_path in tqdm(src_files, desc=f"converting {spk_src}->{spk_tgt}", leave=False):
            out_spk_dir = args.out_dir / spk_src
            out_spk_dir.mkdir(parents=True, exist_ok=True)

            out_path = out_spk_dir / f"{src_path.stem}.wav"

            wav_hat = convert_one(args, wavlm, hifigan, alignments, pick_from_bag, src_path)
            torchaudio.save(str(out_path), wav_hat, 16000)

        del bag, pick_from_bag
        torch.cuda.empty_cache()


if __name__ == "__main__":
    args = check_argv()
    print("Parsed command-line args:")
    for k, v in vars(args).items():
        print(f"  --{k} = {v}")
    main(args)