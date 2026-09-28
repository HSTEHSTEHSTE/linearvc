import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE


def check_argv():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--transforms_path",
        type=Path,
        required=True,
        help="directory that directly contains transforms.npy",
    )

    # first reduction (speaker vectors)
    p.add_argument("--pca_dim", type=int, default=10)

    # 2D viz
    p.add_argument("--method", type=str, default="tsne", choices=["pca", "tsne"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--title", type=str, default="Speaker transforms (2D)")

    # GMM + sampling
    p.add_argument("--gmm_components", type=int, default=4)
    p.add_argument("--gmm_select_trials", type=int, default=10)
    p.add_argument("--gmm_seed_offset", type=int, default=0)
    p.add_argument("--reg_covar", type=float, default=1e-6)
    p.add_argument("--cov_scale", type=float, default=1.0)
    p.add_argument("--k", type=int, default=200, help="number of sampled points to overlay")

    p.add_argument("--out", type=Path, default=None, help="output image path; if omitted, show interactively")
    p.add_argument("--no_labels", action="store_true", help="don’t draw speaker id text labels")
    return p.parse_args()


def fit_pca(X: np.ndarray, out_dim: int):
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Vt[:out_dim]          # (out_dim, D)
    Z = Xc @ P.T              # (N, out_dim)
    return mu, P, Z


def weight_evenness_score(weights: np.ndarray) -> float:
    k = weights.shape[0]
    u = np.full(k, 1.0 / k, dtype=np.float64)
    w = weights.astype(np.float64, copy=False)
    return float(np.linalg.norm(w - u, ord=2))


def fit_best_gmm(Z: np.ndarray, *, n_components: int, trials: int, seed: int, seed_offset: int, reg_covar: float):
    best_gmm = None
    best_score = float("inf")
    best_trial = None

    for t in range(trials):
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            reg_covar=reg_covar,
            random_state=seed + seed_offset + t,
            n_init=10,
            max_iter=500,
            init_params="kmeans",
        )
        gmm.fit(Z)
        score = weight_evenness_score(gmm.weights_)
        if score < best_score:
            best_score = score
            best_gmm = gmm
            best_trial = t

    return best_gmm, best_score, best_trial


def sample_from_gmm(rng: np.random.Generator, gmm: GaussianMixture, *, cov_scale: float):
    comp = int(rng.choice(gmm.n_components, p=gmm.weights_))
    cov = gmm.covariances_[comp] * (cov_scale ** 2)
    z = rng.multivariate_normal(mean=gmm.means_[comp], cov=cov)
    return z, comp


def main(args):
    assert args.pca_dim >= 1, "Meow: --pca_dim must be >= 1."
    assert args.k >= 1, "Meow: --k must be >= 1."
    assert args.gmm_components >= 1, "Meow: --gmm_components must be >= 1."
    assert args.gmm_select_trials >= 1, "Meow: --gmm_select_trials must be >= 1."
    assert args.cov_scale > 0, "Meow: --cov_scale must be > 0."
    assert args.reg_covar >= 0, "Meow: --reg_covar must be >= 0."

    rng = np.random.default_rng(args.seed)

    transforms_file = Path(args.transforms_path) / "transforms.npy"
    if not transforms_file.exists():
        raise FileNotFoundError(f"Meow: missing file: {transforms_file}")

    transforms = np.load(transforms_file, allow_pickle=True).item()
    keys = list(transforms.keys())

    X = np.stack([np.asarray(transforms[k]) for k in keys], axis=0)  # (N, r, feat)
    N, r_dim, feat_dim = X.shape
    Xv = X.reshape(N, r_dim * feat_dim).astype(np.float64)

    # speaker vectors in PCA space (pca_dim), meow
    _, _, Zp = fit_pca(Xv, args.pca_dim)  # (N, pca_dim)

    # ---- Fit tSNE/PCA on real speakers ONLY ----
    if args.method == "pca":
        _, _, Z2_real = fit_pca(Zp, 2)
        embedder = None
    else:
        embedder = TSNE(n_components=2, random_state=args.seed, init="pca")
        Z2_real = embedder.fit_transform(Zp)  # trained only on real speakers, meow

    # ---- Fit GMM on real speakers in PCA space, sample k pseudo speakers ----
    gmm, gmm_score, gmm_trial = fit_best_gmm(
        Zp,
        n_components=args.gmm_components,
        trials=args.gmm_select_trials,
        seed=args.seed,
        seed_offset=args.gmm_seed_offset,
        reg_covar=args.reg_covar,
    )

    Zp_samp = np.stack(
        [sample_from_gmm(rng, gmm, cov_scale=args.cov_scale)[0] for _ in range(args.k)],
        axis=0,
    )  # (k, pca_dim)

    # ---- Map sampled points into the *existing* 2D space ----
    # t-SNE has no reliable out-of-sample transform; so we do:
    # - if method=pca: just project via PCA to 2D (consistent)
    # - if method=tsne: place samples using nearest-neighbor averaging in the REAL t-SNE space, meow
    if args.method == "pca":
        _, _, Z2_samp = fit_pca(Zp_samp, 2)  # PCA on samples alone is NOT consistent, so do consistent projection:
        # Fix: recompute 2D PCA basis from real Zp, then apply to samples
        mu2 = Zp.mean(axis=0, keepdims=True)
        Zc = Zp - mu2
        _, _, Vt2 = np.linalg.svd(Zc, full_matrices=False)
        P2 = Vt2[:2]  # (2, pca_dim)
        Z2_real = (Zp - mu2) @ P2.T
        Z2_samp = (Zp_samp - mu2) @ P2.T
    else:
        # Nearest-neighbor interpolation in PCA space -> tSNE space, meow
        # For each sample, find top-m neighbors among real Zp, then average their tSNE coords weighted by distance.
        m = min(20, N)
        eps = 1e-8
        Z2_samp = np.empty((args.k, 2), dtype=np.float64)
        for i in range(args.k):
            diff = Zp - Zp_samp[i][None, :]
            d2 = np.sum(diff * diff, axis=1)
            nn = np.argpartition(d2, m - 1)[:m]
            w = 1.0 / (np.sqrt(d2[nn]) + eps)
            w = w / np.sum(w)
            Z2_samp[i] = (w[:, None] * Z2_real[nn]).sum(axis=0)

    # ---- Plot ----
    plt.figure(figsize=(8, 7))
    plt.scatter(Z2_real[:, 0], Z2_real[:, 1], s=18, alpha=0.85, label="real speakers")
    plt.scatter(Z2_samp[:, 0], Z2_samp[:, 1], s=14, alpha=0.55, label="GMM samples")

    if not args.no_labels:
        for i, k in enumerate(keys):
            plt.text(Z2_real[i, 0], Z2_real[i, 1], str(k), fontsize=6, alpha=0.6)

    plt.title(
        f"{args.title}\n"
        f"GMM comps={args.gmm_components} trial={gmm_trial} evennessL2={gmm_score:.3g} cov_scale={args.cov_scale:g}"
    )
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.legend(loc="best")
    plt.tight_layout()

    if args.out is None:
        plt.show()
    else:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=200)
        print(f"Meow: wrote plot to {out}")


if __name__ == "__main__":
    main(check_argv())