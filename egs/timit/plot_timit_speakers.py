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
    speaker = str(phn_file).split('/')[-2]
    with open(phn_file, 'r') as phns:
        for line in phns:
            line_elements = line.strip().split(' ')
            start = int(int(line_elements[0]) / 16 / 20) # /16: frame to millisecond; 20: num of milliseconds in each WavLM frame
            end = int(int(line_elements[1]) / 16 / 20)
            phn = line_elements[2]
            frames = content[start:end]
            if start < end:
                if phn != 'h#':
                    if phn not in phoneme_frames:
                        phoneme_frames[phn] = {
                            speaker: list(frames)
                        }
                    else:
                        if speaker not in phoneme_frames:
                            phoneme_frames[phn][speaker] = list(frames)
                        else:
                            phoneme_frames[phn][speaker] += list(frames)

for phn in phoneme_frames:
    for speaker in phoneme_frames[phn]:
        phoneme_frames[phn][speaker] = np.stack(phoneme_frames[phn][speaker], axis=0)
np.save(content_path.parent / 'phonemes_spk.npy', phoneme_frames)

def persistent_hash_string(data_string):
    """Generates a persistent SHA256 hash for a string."""
    return abs(int(hashlib.sha256(data_string.encode('utf-8')).hexdigest(), 16)) % 2**32

seed = persistent_hash_string('timit_experiment')
np.random.seed(seed)

sample_size = 5
num_speakers = 10
samples = []
chosen_phoneme = 'eh'
# speaker_indices = np.random.choice(np.arange(len(list(phoneme_frames[chosen_phoneme].keys()))), size=num_speakers, replace=False)
# speakers = []
# all_speakers = list(phoneme_frames[chosen_phoneme].keys())
# for speaker_index in speaker_indices:
#     speakers.append(all_speakers[speaker_index])
# for speaker in speakers:
#     if len(phoneme_frames[chosen_phoneme][speaker]) < sample_size:
#         sample_size = len(phoneme_frames[chosen_phoneme][speaker])
speakers = ['FAKS0', 'FELC0', 'FJEM0', 'MRES0', 'MPAM0', 'MJLN0', 'MAJC0', 'FJSJ0', 'FCMH1', 'MDBB0']
for speaker in speakers:
    phn_sample_indices = np.random.choice(np.arange(phoneme_frames[chosen_phoneme][speaker].shape[0]), size=sample_size, replace=False)
    samples.append(phoneme_frames[chosen_phoneme][speaker][phn_sample_indices])

# PCA
print("Fitting PCA")
samples = np.concatenate(samples, axis=0)
scaler = StandardScaler()
samples_scaled = scaler.fit_transform(samples)
pca = PCA(n_components=2)
samples_pca = pca.fit_transform(samples)

for speaker_index, speaker in enumerate(speakers):
    points = samples_pca[speaker_index * sample_size: (speaker_index + 1) * sample_size]
    plt.scatter(points[:, 0], points[:, 1], label=speaker)

plt.legend()
plt.show()
plt.savefig(content_path.parent.parent / 'pca.png')
plt.clf()

# t-SNE
print("Fitting t-SNE")
tsne = TSNE(n_components=2, random_state=seed)
tsne_results = tsne.fit_transform(samples)

for speaker_index, speaker in enumerate(speakers):
    points = tsne_results[speaker_index * sample_size: (speaker_index + 1) * sample_size]
    plt.scatter(points[:, 0], points[:, 1], label=speaker)

plt.legend()
plt.show()
plt.savefig(content_path.parent.parent / 'tsne.png')
plt.clf()

# UMAP
print("Fitting UMAP")
umap = UMAP(n_components=2, init='random', random_state=seed)
proj = umap.fit_transform(samples)

for speaker_index, speaker in enumerate(speakers):
    points = proj[speaker_index * sample_size: (speaker_index + 1) * sample_size]
    plt.scatter(points[:, 0], points[:, 1], label=speaker)

plt.legend()
plt.show()
plt.savefig(content_path.parent.parent / 'umap.png')
plt.clf()