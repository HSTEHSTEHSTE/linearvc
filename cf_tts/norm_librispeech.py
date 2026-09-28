from pathlib import Path
from tqdm import tqdm

import torch, torchaudio
from cuvs.neighbors import brute_force

from linearvc import linearvc
from linearvc.cf_tts.utils.common import get_speaker_feats, match_knn

device = 'cuda'


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

feats = get_speaker_feats(
    tgt_speaker_root=Path('/home/hltcoe/xli/ARTS/anon_baseline/data/LibriSpeech/dev-clean/1272'),
    linearvc_model=linearvc_model,
    device=device
)
index = brute_force.build(feats)
transform = {
    'feats': feats,
    'index': index,
    'brute_force': brute_force
}

librispeech_root = Path('/home/hltcoe/xli/ARTS/anon_baseline/data/LibriSpeech')
out_path = Path('/home/hltcoe/xli/ARTS/anon_baseline/data/librispeech_norm')

wavs = list(librispeech_root.rglob("*.flac"))
with torch.no_grad():
    for wav_dir in tqdm(wavs):
        wav, sr = torchaudio.load(wav_dir)
        input_features, _ = linearvc_model.wavlm.extract_features(wav.to(device), output_layer=6)
        input_features = match_knn(input_features, transform)
        out_wav = hifigan(input_features).detach().cpu().squeeze(0)
        out_wav_dir = (out_path / wav_dir.relative_to(librispeech_root))
        out_wav_dir.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save((out_path / wav_dir.relative_to(librispeech_root)), out_wav, 16000)
