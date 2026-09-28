from pathlib import Path
import torch
import torchaudio
from linearvc.linearvc import LinearVC

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
linearvc_model = LinearVC(wavlm, hifigan, num_frames=1, device=device)

# Lists of source and target audio files
source_wavs = Path('/home/hltcoe/xli/ARTS/anon_baseline/data/LibriSpeech/test-clean/61').rglob('*.flac')
target_wavs = Path('/home/hltcoe/xli/ARTS/anon_baseline/data/LibriSpeech/test-clean/121').rglob('*.flac')

# Source input utterance
input_features = linearvc_model.get_features("/home/hltcoe/xli/ARTS/anon_baseline/data/LibriSpeech/test-clean/61/70968/61-70968-0000.flac")

# Voice conversion projection matrix
W = linearvc_model.get_projmat(
    source_wavs,
    target_wavs,
    parallel=False,  # enable if parallel
    vad=False
)

# Project the input and vocode
output_wav = linearvc_model.project_and_vocode(input_features, W)
torchaudio.save("output.wav", output_wav[None], 16000)