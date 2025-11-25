import numpy as np
from pathlib import Path
from tqdm import tqdm

timit_wavlm_feat_path = Path('exp/wavlm_feats/timit/TEST')
speakers = sorted([p.stem for p in (timit_wavlm_feat_path / 'spks').iterdir()])
timit_wavlm_feats = list((timit_wavlm_feat_path / 'utts').rglob('*.npy'))


speaker_matrix_path = Path('/home/hltcoe/xli/ARTS/linearvc/exp/content_factorization/TIMIT_TEST/spk_0_r10/VT.npy')
speaker_matrix = np.load(speaker_matrix_path)
out_path = speaker_matrix_path.parent / 'content'

print("Reshaping")
transforms = {
    f"{speaker}": np.linalg.pinv(speaker_matrix[i, :, :]) for i, speaker in enumerate(speakers)
}

for timit_wavlm_feat in tqdm(timit_wavlm_feats):
    feat = np.load(timit_wavlm_feat)
    speaker = str(timit_wavlm_feat).split('/')[-2]
    content = feat @ transforms[speaker]
    output_fn = (out_path / (timit_wavlm_feat.parent.relative_to((timit_wavlm_feat_path / 'utts'))) / timit_wavlm_feat.stem).with_suffix(".npy")
    output_fn.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_fn, content)
