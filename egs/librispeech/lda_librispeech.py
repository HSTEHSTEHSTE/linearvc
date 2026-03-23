import argparse
import math
from pathlib import Path
import pickle
from tqdm import tqdm
import numpy as np
import torch, torchaudio
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from datasets import load_dataset
from linearvc.randomized_lda import RandomizedLDA

device = "cuda"

def check_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subsets",
        type=str,
        nargs='+',
        help="LibriSpeech subsets used",
    )
    parser.add_argument(
        "--rank",
        type=int,
        help="LDA rank",
        default=50
    )
    parser.add_argument(
        "--target_root",
        type=Path,
        help="Root path to output LDA file",
    )
    return parser.parse_args()

def main(args):
    wavlm = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, device=device)
    hifigan, _ = torch.hub.load("bshall/knn-vc", "hifigan_wavlm", trust_repo=True, device=device, prematched=True)

    subsets = args.subsets
    rank = args.rank
    lda_type = 'lda'

    target_file = Path(args.target_root / str(rank) / 'lda.pkl')
    if not target_file.is_file():
        # collect phoneme alignments
        phoneme_frames = {}
        ds = load_dataset("gilkeyio/librispeech-alignments")
        for subset in subsets:
            for item in tqdm(ds[subset]):
                wav = item['audio'].get_all_samples().data.to(device)
                sr = item['audio'].get_all_samples().sample_rate
                with torch.no_grad():
                    x, _ = wavlm.extract_features(wav, output_layer=6)
                x = x.squeeze(0).detach().cpu().numpy()
                current_frame = 0
                for phoneme_info in item['phonemes']:
                    phoneme = phoneme_info['phoneme']
                    start = math.floor(phoneme_info['start'] * 50)
                    end = math.ceil(phoneme_info['end'] * 50)
                    if start >= end:
                        continue
                    if start > current_frame:
                        silence_frames = list(x[current_frame:start])
                        phoneme_frames.setdefault('sil', []).extend(silence_frames)
                    frames = list(x[start:end])
                    phoneme_frames.setdefault(phoneme, []).extend(frames)
                    current_frame = end
        for phn in phoneme_frames:
            phoneme_frames[phn] = np.array(phoneme_frames[phn])

        X = []
        y = []

        for label, feats in phoneme_frames.items():
            X.append(feats)
            y.append(np.full(len(feats), label))

        X = np.vstack(X)   # shape (N, d)
        y = np.concatenate(y)  # shape (N,)

        lda = RandomizedLDA(n_components=rank)
        lda.fit(X, y)

        target_file.parent.mkdir(parents=True, exist_ok=True)
        with open(target_file, 'wb') as file:
            pickle.dump(lda, file)
    else:
        with open(target_file, 'rb') as file:
            lda = pickle.load(file)

if __name__ == "__main__":
    args = check_argv()
    main(args)
