from pathlib import Path
import pickle
from tqdm import tqdm
import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from linearvc.utils import collect_phoneme_frames

device = "cuda"
wavlm = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, device=device)
hifigan, _ = torch.hub.load("bshall/knn-vc", "hifigan_wavlm", trust_repo=True, device=device, prematched=True)

subset = "TRAIN"
rank = 50
wav_dir = Path(f"/home/hltcoe/xli/ARTS/corpora/TIMIT/TIMIT/{subset}")

feats_dir_root = Path(f"/home/hltcoe/xli/ARTS/linearvc/exp/wavlm_feats/timit/")
feats_dir = Path(f"/home/hltcoe/xli/ARTS/linearvc/exp/wavlm_feats/timit/{subset}_vc/spks")
feats_dict = {}
print("Reading from:", feats_dir)
for speaker_feats_fn in tqdm(sorted(feats_dir.glob("*.npy"))):
    speaker = speaker_feats_fn.stem
    feats_dict[speaker] = np.load(speaker_feats_fn, allow_pickle=True)
print("No. speakers:", len(feats_dict))

XS = []
speakers = sorted(feats_dict)
for speaker in speakers:
    XS.append(feats_dict[speaker][:, :])


phoneme_frames = collect_phoneme_frames(feats_dir_root / subset, wav_dir, 'wavlm')

phoneme_frames_phones = {}
for phn in phoneme_frames:
    frames = []
    for speaker in phoneme_frames[phn]:
        frames.append(phoneme_frames[phn][speaker])
    phoneme_frames_phones[phn] = np.concatenate(frames)
phoneme_frames_speakers = {}
for phn in phoneme_frames:
    for speaker in phoneme_frames[phn]:
        phoneme_frames_speakers.setdefault(speaker, []).append(phoneme_frames[phn][speaker])
for speaker in phoneme_frames_speakers:
    phoneme_frames_speakers[speaker] = np.concatenate(phoneme_frames_speakers[speaker])

target_file = Path('/home/hltcoe/xli/ARTS/linearvc/exp/lda/TIMIT/' + str(rank) + '/lda.pkl')
if not target_file.is_file():
    X = []
    y = []

    for label, feats in phoneme_frames_phones.items():
        X.append(feats)
        y.append(np.full(len(feats), label))

    X = np.vstack(X)   # shape (N, d)
    y = np.concatenate(y)  # shape (N,)

    lda = LinearDiscriminantAnalysis(n_components=rank)
    lda.fit(X, y)

    target_file.parent.mkdir(parents=True, exist_ok=True)
    with open(target_file, 'wb') as file:
        pickle.dump(lda, file)
else:
    with open(target_file, 'rb') as file:
        lda = pickle.load(file)
