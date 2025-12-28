import numpy as np
import torch, torchaudio
from linearvc import linearvc
from pathlib import Path

src_spk = Path('/home/hltcoe/xli/ARTS/corpora/TIMIT/TIMIT/TEST/DR1/FAKS0')
# tgt_spk = Path('/home/hltcoe/xli/ARTS/corpora/TIMIT/TIMIT/TEST/DR2/MABW0')
tgt_spk = Path('/home/hltcoe/xli/ARTS/corpora/TIMIT/TIMIT/TRAIN/DR2/MARC0')
# other_spk = Path('/home/hltcoe/xli/ARTS/corpora/TIMIT/TIMIT/TEST/DR2/FCMR0')
other_spk = Path('/home/hltcoe/xli/ARTS/corpora/TIMIT/TIMIT/TRAIN/DR2/MPPC0')

convert_wav = 'SA1'
# input_wav = src_spk / (convert_wav + '.WAV')
input_wav = Path('/home/hltcoe/xli/ARTS/anon_baseline/data/LibriSpeech/dev-clean/84/121123/84-121123-0001.flac')
content_path = Path('/home/hltcoe/xli/ARTS/linearvc/exp/content_factorization/TIMIT_TRAIN_vc/spk_0_r100')
out_path = Path('/home/hltcoe/xli/ARTS/linearvc/exp/vc/TIMIT')

vc_type = 'linear' # linear, content

device = "cuda"  # "cpu"

# Load all the required models
wavlm = torch.hub.load(
    "bshall/knn-vc", 
    "wavlm_large", 
    trust_repo=True, 
    progress=True, 
    device=device, 
)
hifigan, _ = torch.hub.load(
    "bshall/knn-vc",
    "hifigan_wavlm",
    trust_repo=True,
    prematched=True,
    progress=True,
    device=device,
)
linearvc_model = linearvc.LinearVC(wavlm, hifigan, device)
input_features = linearvc_model.get_features(input_wav)

def get_wavs(spk_path):
    wavs_all = list(spk_path.rglob('*.WAV'))
    wavs = []
    for wav in wavs_all:
        if not wav.stem == convert_wav:
            wavs.append(wav)
    return wavs

if vc_type == 'linear':
    src_wavs = get_wavs(src_spk)
    tgt_wavs = get_wavs(tgt_spk)

    # Voice conversion projection matrix
    W = linearvc_model.get_projmat(
        src_wavs,
        tgt_wavs,
        parallel=True,  # enable if parallel
        vad=False,
    )

    # Project the input and vocode
    output_wav = linearvc_model.project_and_vocode(input_features, W)
    torchaudio.save(out_path / (convert_wav + '.wav'), output_wav[None], 16000)

elif vc_type == 'content':
    transforms = np.load(content_path / 'transforms.npy', allow_pickle=True).item()
    src_spk_name = str(src_spk).split('/')[-1]
    tgt_spk_name = str(tgt_spk).split('/')[-1]
    other_spk_name = str(other_spk).split('/')[-1]

    input_features = input_features.cpu().numpy()
    # out_features = torch.tensor(np.dot(np.dot(input_features, np.linalg.pinv(transforms[src_spk_name])), transforms[tgt_spk_name])).to(device)
    out_features = torch.tensor(np.dot(np.dot(input_features, np.linalg.pinv(transforms[other_spk_name])), transforms[tgt_spk_name])).to(device)
    wav_hat = hifigan(out_features.unsqueeze(0)).squeeze(0).detach().cpu()

    torchaudio.save(out_path / (convert_wav + '.wav'), wav_hat, 16000)