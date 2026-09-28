from pathlib import Path
import numpy as np
import torch
import torchaudio

from linearvc.linearvc import LinearVC  # meow

# -------------------------
# Device
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Paths / constants
# -------------------------
transforms_npy = Path(
    "/home/hltcoe/xli/ARTS/linearvc/exp/interspeech/cf/transforms/0/rank_75/transforms.npy"
)

wav_path = Path("/home/hltcoe/xli/ARTS/anon_baseline/data/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac")
source_spk = "1089"

out_root = Path("/home/hltcoe/xli/ARTS/linearvc/exp/interspeech/cf/new_speakers")

num_samples = 20
rng = np.random.default_rng(0)

# -------------------------
# Load transforms (numpy)
# -------------------------
transforms = np.load(transforms_npy, allow_pickle=True).item()  # dict: spk -> (75,1024)
keys = list(transforms.keys())
K = len(keys)

assert source_spk in transforms, f"Meow! {source_spk} not found in transforms keys."

X = np.stack([np.asarray(transforms[k]) for k in keys], axis=0)  # (K,75,1024)
K, R, F = X.shape
D = R * F
Xv = X.reshape(K, D).astype(np.float64)

# -------------------------
# Output dirs
# -------------------------
dirs = {
    "pair_linear_combo": out_root / "pair_linear_combo",
    "random_transform_sanity": out_root / "random_transform_sanity",
    "all_speakers_linear_combo": out_root / "all_speakers_linear_combo",
    "convex_hull_sample": out_root / "convex_hull_sample",
    "gaussian_pca10": out_root / "gaussian_pca10",
    "gaussian_pca20": out_root / "gaussian_pca20",
    "gaussian_fullrank_diag": out_root / "gaussian_fullrank_diag",
}
for d in dirs.values():
    d.mkdir(parents=True, exist_ok=True)

# -------------------------
# Sampling helpers
# -------------------------
def save_transform_and_meta(out_dir: Path, sid: str, T: np.ndarray, meta: str = ""):
    np.save(out_dir / f"{sid}.npy", T)
    if meta:
        with open(out_dir / f"{sid}.txt", "w") as f:
            f.write(meta)

def sample_pairwise(num: int):
    samples = []
    for i in range(num):
        sid = f"{i:04d}"
        ia, ib = rng.choice(K, size=2, replace=False)
        spk_a, spk_b = keys[ia], keys[ib]
        Ta = np.asarray(transforms[spk_a])
        Tb = np.asarray(transforms[spk_b])
        alpha = rng.random()
        T = (alpha * Ta + (1.0 - alpha) * Tb).astype(Ta.dtype, copy=False)
        meta = f"spk_a={spk_a}\nspk_b={spk_b}\nalpha={alpha}\n"
        samples.append((sid, T, meta))
    return samples

def sample_random_existing(num: int):
    samples = []
    for i in range(num):
        sid = f"{i:04d}"
        idx = rng.integers(0, K)
        spk = keys[idx]
        T = np.asarray(transforms[spk]).copy()
        meta = f"speaker={spk}\n"
        samples.append((sid, T, meta))
    return samples

def sample_dirichlet_convex(num: int):
    samples = []
    for i in range(num):
        sid = f"{i:04d}"
        w = rng.dirichlet(np.ones(K, dtype=np.float64))     # convex weights
        T = np.tensordot(w, X, axes=(0, 0)).astype(X.dtype, copy=False)
        # save weights too for reproducibility
        meta = f"dirichlet_weights_saved_in={sid}_weights.npy\n"
        samples.append((sid, T, meta, w))
    return samples

def fit_pca(Xv_: np.ndarray, pca_dim: int):
    mu_x = Xv_.mean(axis=0, keepdims=True)
    Xc = Xv_ - mu_x
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Vt[:pca_dim]  # (pca_dim, D)
    Z = Xc @ P.T      # (K, pca_dim)
    return mu_x, P, Z

def sample_gaussian_in_pca(num: int, pca_dim: int):
    mu_x, P, Z = fit_pca(Xv, pca_dim)
    mu_z = Z.mean(axis=0)
    cov_z = np.cov(Z, rowvar=False) + 1e-6 * np.eye(pca_dim)
    Zs = rng.multivariate_normal(mu_z, cov_z, size=num)    # (num, pca_dim)
    Xs = mu_x + (Zs @ P)                                   # (num, D)
    samples = []
    for i in range(num):
        sid = f"{i:04d}"
        T = Xs[i].reshape(R, F).astype(X.dtype, copy=False)
        meta = f"pca_dim={pca_dim}\n"
        samples.append((sid, T, meta))
    return samples

def sample_gaussian_diag_fullspace(num: int):
    # "full rank" in the sense: samples span all D dims, but covariance is diagonal (stable with 40 pts)
    mu = Xv.mean(axis=0)
    var = Xv.var(axis=0, ddof=1) + 1e-6
    std = np.sqrt(var)
    samples = []
    for i in range(num):
        sid = f"{i:04d}"
        z = rng.standard_normal(D)
        Tv = (mu + std * z)
        T = Tv.reshape(R, F).astype(X.dtype, copy=False)
        meta = "gaussian=diag_fullspace\n"
        samples.append((sid, T, meta))
    return samples

# -------------------------
# Generate sampled transforms for each method
# -------------------------
pair_samples = sample_pairwise(num_samples)
rand_samples = sample_random_existing(num_samples)

all_combo_samples = sample_dirichlet_convex(num_samples)      # general all-speaker convex combo
convex_hull_samples = sample_dirichlet_convex(num_samples)    # explicitly labeled convex hull set

g_pca10_samples = sample_gaussian_in_pca(num_samples, pca_dim=10)
g_pca20_samples = sample_gaussian_in_pca(num_samples, pca_dim=20)
g_diag_samples = sample_gaussian_diag_fullspace(num_samples)

# Save npy (+meta)
for sid, T, meta in pair_samples:
    save_transform_and_meta(dirs["pair_linear_combo"], sid, T, meta)

for sid, T, meta in rand_samples:
    save_transform_and_meta(dirs["random_transform_sanity"], sid, T, meta)

for sid, T, meta, w in all_combo_samples:
    np.save(dirs["all_speakers_linear_combo"] / f"{sid}_weights.npy", w)
    save_transform_and_meta(dirs["all_speakers_linear_combo"], sid, T, meta)

for sid, T, meta, w in convex_hull_samples:
    np.save(dirs["convex_hull_sample"] / f"{sid}_weights.npy", w)
    save_transform_and_meta(dirs["convex_hull_sample"], sid, T, meta)

for sid, T, meta in g_pca10_samples:
    save_transform_and_meta(dirs["gaussian_pca10"], sid, T, meta)

for sid, T, meta in g_pca20_samples:
    save_transform_and_meta(dirs["gaussian_pca20"], sid, T, meta)

for sid, T, meta in g_diag_samples:
    save_transform_and_meta(dirs["gaussian_fullrank_diag"], sid, T, meta)

# -------------------------
# Load models like your reference pipeline (LinearVC)
# -------------------------
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
linearvc_model = LinearVC(wavlm, hifigan, device)

# -------------------------
# Extract features using LinearVC (matches your working script)
# -------------------------
with torch.no_grad():
    input_features = linearvc_model.get_features(str(wav_path))  # torch tensor
    input_features = input_features.squeeze(0)                   # (frames,1024)
    input_features_np = input_features.detach().cpu().numpy()

# Precompute pinv of source transform (numpy)
T_src = np.asarray(transforms[source_spk])     # (75,1024)
pinv_T_src = np.linalg.pinv(T_src)             # (1024,75)

# -------------------------
# Synthesis helper (imitate your script)
# -------------------------
@torch.no_grad()
def synth_and_save_from_T(T_tgt_np: np.ndarray, out_wav_path: Path):
    out_features_np = np.dot(np.dot(input_features_np, pinv_T_src), T_tgt_np)  # (frames,1024)
    out_features = torch.tensor(out_features_np).to(device).float()
    wav_hat = hifigan(out_features.unsqueeze(0)).squeeze(0).detach().cpu()

    if wav_hat.dim() == 1:
        wav_hat = wav_hat.unsqueeze(0)  # (1,T)
    torchaudio.save(str(out_wav_path), wav_hat, 16000)

def render_set(samples, out_dir: Path):
    for sid, T, *_ in samples:
        synth_and_save_from_T(T, out_dir / f"{sid}.wav")

# Render audio for each method
render_set(pair_samples, dirs["pair_linear_combo"])
render_set(rand_samples, dirs["random_transform_sanity"])
render_set(all_combo_samples, dirs["all_speakers_linear_combo"])
render_set(convex_hull_samples, dirs["convex_hull_sample"])
render_set(g_pca10_samples, dirs["gaussian_pca10"])
render_set(g_pca20_samples, dirs["gaussian_pca20"])
render_set(g_diag_samples, dirs["gaussian_fullrank_diag"])

print("Meow! Done. Outputs:")
for name, d in dirs.items():
    print(f"  {name}: {d}")