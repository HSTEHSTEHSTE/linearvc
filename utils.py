"""
Miscellaneous utility functions.

Author: Matthew Baas, Benjamin van Niekerk, Herman Kamper, Henry Li Xinyuan
Date: 2026
"""

from pathlib import Path
from tqdm import tqdm
import numpy as np
from torch import Tensor
import torch
import torch.nn.functional as F


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fast_cosine_dist(
    source_feats: Tensor, matching_pool: Tensor, device: str = "cpu"
) -> Tensor:
    """
    Like torch.cdist, but fixed dim=-1 and for cosine distance.

    Based on:
    <https://github.com/bshall/knn-vc/blob/master/matcher.py>
    """
    source_norms = torch.norm(source_feats, p=2, dim=-1).to(device)
    matching_norms = torch.norm(matching_pool, p=2, dim=-1)
    dotprod = (
        -torch.cdist(source_feats[None].to(device), matching_pool[None], p=2)[0]
        ** 2
        + source_norms[:, None] ** 2
        + matching_norms[None] ** 2
    )
    dotprod /= 2

    dists = 1 - (dotprod / (source_norms[:, None] * matching_norms[None]))
    return dists


def pca_transform(
    X: Tensor, mean: Tensor, components: Tensor, explained_variance: Tensor
) -> Tensor:
    X = X - mean
    X_transformed = X @ components.T
    X_transformed /= torch.sqrt(explained_variance)
    return X_transformed


def pca_inverse_transform(
    X: Tensor, mean: Tensor, components: Tensor, explained_variance: Tensor
) -> Tensor:
    Xi = X @ (explained_variance[:, None].sqrt() * components)
    return Xi + mean

def stack_frames(x, k):
    # x: [n, d]
    n, d = x.shape

    if k > n:
        raise ValueError("k cannot be larger than n")

    # [n-k+1, k, d]
    windows = x.unfold(0, k, 1)

    # reshape → [n-k+1, k*d]
    return windows.contiguous().view(n - k + 1, k * d)

def unwrap_frames(y, k):
    # y: [n-k+1, k*d]
    n_k1, kd = y.shape
    d = kd // k
    n = n_k1 + k - 1

    # reshape to match fold expectations
    # treat d as channels
    windows = y.view(n_k1, k, d).permute(2, 1, 0)  # [d, k, n-k+1]
    windows = windows.reshape(1, d * k, n_k1)      # [1, d*k, L]

    # overlap-add
    out = F.fold(
        windows,
        output_size=(1, n),
        kernel_size=(1, k),
        stride=(1, 1)
    )  # [1, d, 1, n]

    # compute counts the same way (all ones)
    ones = torch.ones_like(windows)
    counts = F.fold(
        ones,
        output_size=(1, n),
        kernel_size=(1, k),
        stride=(1, 1)
    )  # [1, d, 1, n]

    out = out / counts
    return out.squeeze(0).squeeze(1).transpose(0, 1)  # [n, d]

def collect_phoneme_frames(content_path_root, timit_path_root, feature_type):
    content_path = Path(content_path_root)
    timit_path = Path(timit_path_root)
    phoneme_frames = {}

    for content_file in tqdm(list(content_path.rglob('*.npy'))):
        # skip non-utt files for certain feature types
        if feature_type in ['wavlm', 'contentvec'] and content_file.parts[-4] != 'utts':
            continue

        content = np.load(content_file, allow_pickle=True)

        # resolve PHN path
        if feature_type in ['wavlm', 'contentvec']:
            rel = content_file.relative_to(content_path / 'utts')
        else:  # 'cf'
            rel = content_file.relative_to(content_path)

        phn_file = timit_path / rel.parent / (content_file.stem + '.PHN')
        speaker = phn_file.parent.name

        with open(phn_file, 'r') as f:
            for line in f:
                start, end, phn = line.strip().split()
                start = int(int(start) / 16 / 20)
                end = int(int(end) / 16 / 20)

                if start >= end or phn == 'h#':
                    continue

                frames = list(content[start:end])

                phoneme_frames.setdefault(phn, {}) \
                    .setdefault(speaker, []) \
                    .extend(frames)

    for phn in phoneme_frames:
        for speaker in phoneme_frames[phn]:
            phoneme_frames[phn][speaker] = np.array(phoneme_frames[phn][speaker])
    return phoneme_frames