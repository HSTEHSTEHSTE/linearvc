from numpy import linalg
from pathlib import Path
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm
import argparse
import celer
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sys
import time
import torch
import torchaudio

from linearvc.utils import fast_cosine_dist

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def check_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subset",
        type=str,
        default='dev-clean'
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=8192
    )
    parser.add_argument(
        "--k_top",
        type=int,
        default=4
    )
    parser.add_argument(
        "--feats_dir",
        type=Path,
        help="source speech directory",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=100
    )
    return parser.parse_args()

def align(src, refs):
    neighbors = NearestNeighbors(n_neighbors=1, metric="cosine")
    neighbors.fit(refs)
    dists, indices = neighbors.kneighbors(src)
    return refs[indices.squeeze(), :]

def main(args):
    wavlm = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, device=device)
    hifigan, _ = torch.hub.load("bshall/knn-vc", "hifigan_wavlm", trust_repo=True, device=device, prematched=True)

    subset = args.subset
    n_frames = args.n_frames
    k_top = args.k_top

    feats_dir = Path(args.feats_dir)
    feats_dict = {}
    print("Reading from:", feats_dir)
    for speaker_feats_fn in tqdm(sorted(feats_dir.glob("*.npy"))):
        speaker = speaker_feats_fn.stem
        feats_dict[speaker] = np.load(speaker_feats_fn)
    print("No. speakers:", len(feats_dict))

    rank = args.rank
    XS = []
    speakers = sorted(feats_dict)
    for speaker in speakers:
        XS.append(feats_dict[speaker][:, :])

    for src_spk_index, src_spk in tqdm(enumerate(speakers), total=len(speakers)):
        print("Matching:")
        XS = [align(XS[src_spk_index], X) for X in tqdm(XS)]
        XS = np.concatenate(XS, axis=-1)
        XS = np.float32(XS)

        start_time = time.time()
        print("SVD")
        U, S, VT = randomized_svd(XS, n_components=rank)
        print("Time expired: ", time.time() - start_time)

        print("Reshaping")
        VT = VT.reshape(-1, len(speakers), 1024).swapaxes(0, 1)
        transforms = {
            f"{speaker}": VT[i, :, :] for i, speaker in enumerate(speakers)
        }

        print("Projection matrices:")
        projmats = {}
        for source in tqdm(feats_dict, leave=False):
            for target in feats_dict:
                if source == target:
                    continue
                W = np.linalg.pinv(transforms[source]) @ transforms[target]
                projmats[f"{source}-{target}"] = (W, None)

        out_path = Path('/home/hltcoe/xli/ARTS/linearvc/exp/content_factorization/librispeech_' + subset + '/rank_' + str(rank) + '/src_' + src_spk)
        np.save(out_path / 'XS.npy', XS)
        np.save(out_path / 'U.npy', U)
        np.save(out_path / 'S.npy', S)
        np.save(out_path / 'VT.npy', VT)
        np.save(out_path / 'transforms.npy', transforms)

if __name__ == "__main__":
    args = check_argv()
    main(args)