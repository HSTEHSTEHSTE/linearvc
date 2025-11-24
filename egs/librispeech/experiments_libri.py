from numpy import linalg
from pathlib import Path
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm
import celer
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sys
import time
import torch
import torchaudio

from utils import fast_cosine_dist

device = "cuda"
wavlm = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, device=device)
hifigan, _ = torch.hub.load("bshall/knn-vc", "hifigan_wavlm", trust_repo=True, device=device, prematched=True)

subset = "dev-clean"
wav_dir = Path(f"/home/kamperh/endgame/datasets/librispeech/LibriSpeech/{subset}")
n_frames = 8192
k_top = 4

feats_dir = Path(f"/home/hltcoe/xli/ARTS/linearvc/exp/wavlm_feats/librispeech/{subset}")
feats_dict = {}
print("Reading from:", feats_dir)
for speaker_feats_fn in tqdm(sorted(feats_dir.glob("*.npy"))):
    speaker = speaker_feats_fn.stem
    feats_dict[speaker] = np.load(speaker_feats_fn)
print("No. speakers:", len(feats_dict))

rank = 100
XS = []
speakers = sorted(feats_dict)
for speaker in speakers:
    XS.append(feats_dict[speaker][:, :])

def align(src, refs):
    neighbors = NearestNeighbors(n_neighbors=1, metric="cosine")
    neighbors.fit(refs)
    dists, indices = neighbors.kneighbors(src)
    return refs[indices.squeeze(), :]

print("Matching:")
XS = [align(XS[0], X) for X in tqdm(XS)]
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
for source in tqdm(feats_dict):
    for target in tqdm(feats_dict, leave=False):
        if source == target:
            continue
        W = np.linalg.pinv(transforms[source]) @ transforms[target]
        projmats[f"{source}-{target}"] = (W, None)

out_path = Path('/home/hltcoe/xli/ARTS/linearvc/exp/speaker_transforms/librispeech_' + subset + '/src_1272')
np.save(out_path / 'XS.npy', XS)
np.save(out_path / 'U.npy', U)
np.save(out_path / 'S.npy', S)
np.save(out_path / 'VT.npy', VT)
