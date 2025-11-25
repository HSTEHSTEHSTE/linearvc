import hashlib
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

content_path = Path('/home/hltcoe/xli/ARTS/linearvc/exp/content_factorization/TIMIT_TEST/spk_0_r10/content')
# content_path = Path('/home/hltcoe/xli/ARTS/linearvc/exp/wavlm_feats/timit/TEST/utts')
timit_path = Path('/home/hltcoe/xli/ARTS/corpora/TIMIT/TIMIT/TEST')

phoneme_frames = {}
content_files = list(content_path.rglob('*.npy'))

for content_file in tqdm(content_files):
    content = np.load(content_file)
    phn_file = timit_path / (content_file.relative_to(content_path)).parent / (content_file.stem + '.PHN')
    with open(phn_file, 'r') as phns:
        for line in phns:
            line_elements = line.strip().split(' ')
            start = int(int(line_elements[0]) / 16 / 20) # /16: frame to millisecond; 20: num of milliseconds in each WavLM frame
            end = int(int(line_elements[1]) / 16 / 20)
            phn = line_elements[2]
            frames = content[start:end]
            if phn != 'h#':
                if phn not in phoneme_frames:
                    phoneme_frames[phn] = list(frames)
                else:
                    phoneme_frames[phn] += list(frames)

for phn in phoneme_frames:
    phoneme_frames[phn] = np.stack(phoneme_frames[phn], axis=0)
np.save(content_path.parent / 'phonemes.npy', phoneme_frames)

def persistent_hash_string(data_string):
    """Generates a persistent SHA256 hash for a string."""
    return abs(int(hashlib.sha256(data_string.encode('utf-8')).hexdigest(), 16)) % 2**32

seed = persistent_hash_string('timit_experiment')
np.random.seed(seed)

frame_num = np.inf
for phn in phoneme_frames:
    if len(phoneme_frames[phn]) < frame_num:
        frame_num = len(phoneme_frames[phn])
sample_size = min(frame_num, 100)
samples = []
for phn in phoneme_frames:
    phn_sample_indices = np.random.choice(np.arange(phoneme_frames[phn].shape[0]), size=sample_size, replace=False)
    samples.append(phoneme_frames[phn][phn_sample_indices])

# PCA
print("Fitting PCA")
samples = np.concatenate(samples, axis=0)
scaler = StandardScaler()
samples_scaled = scaler.fit_transform(samples)
pca = PCA(n_components=2)
samples_pca = pca.fit_transform(samples)

for phn_index, phn in enumerate(phoneme_frames):
    if phn_index >= 10:
        break
    points = samples_pca[phn_index * sample_size: (phn_index + 1) * sample_size]
    plt.scatter(points[:, 0], points[:, 1], label=phn)

plt.legend()
plt.show()
plt.savefig(content_path.parent.parent / 'pca.png')
plt.clf()

# t-SNE
print("Fitting t-SNE")
tsne = TSNE(n_components=2, random_state=seed)
tsne_results = tsne.fit_transform(samples)

for phn_index, phn in enumerate(phoneme_frames):
    if phn_index >= 10:
        break
    points = tsne_results[phn_index * sample_size: (phn_index + 1) * sample_size]
    plt.scatter(points[:, 0], points[:, 1], label=phn)

plt.legend()
plt.show()
plt.savefig(content_path.parent.parent / 'tsne.png')
plt.clf()

# UMAP
print("Fitting UMAP")
umap = UMAP(n_components=2, init='random', random_state=seed)
proj = umap.fit_transform(samples)

for phn_index, phn in enumerate(phoneme_frames):
    if phn_index >= 10:
        break
    points = proj[phn_index * sample_size: (phn_index + 1) * sample_size]
    plt.scatter(points[:, 0], points[:, 1], label=phn)

plt.legend()
plt.show()
plt.savefig(content_path.parent.parent / 'umap.png')
plt.clf()