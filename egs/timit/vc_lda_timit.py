import pickle
from pathlib import Path
import numpy as np
import torch, torchaudio

device = 'cuda'

# prepare models
with open('/home/hltcoe/xli/ARTS/linearvc/exp/lda/TIMIT/59/lda.pkl', 'rb') as file:
    lda = pickle.load(file)

wavlm = torch.hub.load(
    "bshall/knn-vc", "wavlm_large", trust_repo=True, device=device
)

hifigan, _ = torch.hub.load(
    "bshall/knn-vc",
    "hifigan_wavlm",
    trust_repo=True,
    prematched=True,
    progress=True,
    device=device,
)

new_speaker_path = Path('/home/hltcoe/xli/ARTS/linearvc/exp/wavlm_feats/librispeech/test-clean/61.npy')

input_utterance = Path('/home/hltcoe/xli/ARTS/anon_baseline/data/LibriSpeech/test-clean/61/70968/61-70968-0000.flac')
wav, sr = torchaudio.load(str(input_utterance))
with torch.inference_mode():
    x, _ = wavlm.extract_features(wav.to(device), output_layer=6)

new_speaker_feats = np.load(new_speaker_path)
new_speaker_contents = np.matmul(new_speaker_feats, lda.scalings_)
uscf = torch.tensor(lda.scalings_).to(device)
W = torch.tensor(np.matmul(np.linalg.pinv(new_speaker_contents), new_speaker_feats)).to(device)
new_feats = torch.matmul(torch.matmul(x, uscf), W)
wav_hat = hifigan(new_feats).squeeze(0).detach().cpu()

breakpoint()