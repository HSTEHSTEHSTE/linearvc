import argparse
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
        choices=["mean", "dtw"],
        help="mean, dtw",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help="mean pooled length scaling",
        default=1.,
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
    return parser.parse_args()

def length_scaled_pool(X, alpha):
    n = X.shape[0]
    return X.mean(axis=0) * (n ** alpha)

def dtw(X, Y, lam=0.0):
    n, m = len(X), len(Y)

    # Precompute all pairwise cosine distances
    # X: [n, d], Y: [m, d] -> [n, m]
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

def main(args):
    wavlm = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, device=device)
    hifigan, _ = torch.hub.load("bshall/knn-vc", "hifigan_wavlm", trust_repo=True, device=device, prematched=True)
    alpha = args.alpha
    lam = args.lam

    src_wav_path = Path('/home/hltcoe/xli/ARTS/anon_baseline/data/LibriSpeech/test-clean/61/70968/61-70968-0000.flac')
    tgt_speaker_path = Path('/home/hltcoe/xli/ARTS/anon_baseline/data/LibriSpeech/test-clean/121')
    tgt_speaker_wavs = list(tgt_speaker_path.rglob("*.flac"))

    with open('/home/hltcoe/xli/ARTS/linearvc/exp/asr/LibriSpeech/alignments/alignments.json', 'r') as file:
        alignments = json.load(file)

    bag = []
    for tgt_speaker_wav in tqdm(tgt_speaker_wavs):
        alignment = alignments[tgt_speaker_wav.stem]
        wav, sr = torchaudio.load(tgt_speaker_wav)
        with torch.no_grad():
            x, _ = wavlm.extract_features(wav.to(device), output_layer=6)
        x = x.squeeze(0)
        current_frame = 0
        for phoneme_info in alignment:
            phoneme = phoneme_info['phoneme']
            start = math.floor(phoneme_info['start'] * 50)
            end = math.ceil(phoneme_info['end'] * 50)
            if start > current_frame:
                silence_frames = x[current_frame:start]
                if args.bag_silence:
                    bag.append(silence_frames)
            if start >= end:
                continue
            frames = x[start:end]
            current_frame = end
            bag.append(frames)
    
    # -----------------------------------------
    # Precompute pooled means if using "mean"
    # -----------------------------------------
    bag_means = None
    if args.unit_distance_type == "mean":
        with torch.no_grad():
            bag_means = torch.stack([length_scaled_pool(seg, alpha) for seg in bag], dim=0)  # [B, D]
            bag_means = bag_means.to(device)
    
    # -----------------------------------------
    # Main unit selection function
    # -----------------------------------------
    def pick_from_bag(query_frames):
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

        else:
            raise ValueError(f"Unknown unit_distance_type: {args.unit_distance_type}")

    # ----------------------------
    # Process source and replace
    # ----------------------------
    wav, sr = torchaudio.load(src_wav_path)
    with torch.no_grad():
        x, _ = wavlm.extract_features(wav.to(device), output_layer=6)
    x = x.squeeze(0)

    alignment = alignments[src_wav_path.stem]
    new_seq = []
    current_frame = 0

    for phoneme_info in tqdm(alignment):
        start = math.floor(phoneme_info['start'] * 50)
        end = math.ceil(phoneme_info['end'] * 50)

        if start > current_frame:
            silence_frames = x[current_frame:start]
            if args.bag_silence:
                new_seq.append(pick_from_bag(silence_frames))
            else:
                new_seq.append(silence_frames)

        if start >= end:
            continue

        frames = x[start:end]
        new_seq.append(pick_from_bag(frames))
        current_frame = end

    new_seq = torch.cat(new_seq, axis=0)
    with torch.no_grad():
        wav_hat = hifigan(new_seq.unsqueeze(0)).squeeze(0).detach().cpu()
    torchaudio.save('output.wav', wav_hat, 16000)

if __name__ == "__main__":
    args = check_argv()
    main(args)
