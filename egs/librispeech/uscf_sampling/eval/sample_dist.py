import argparse
import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm
from sklearn.mixture import GaussianMixture


def check_argv():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--transforms_path",
        type=Path,
        required=True,
        help="directory that directly contains transforms.npy",
    )
    p.add_argument("--pca_dim", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)

    # n/k controls
    p.add_argument("--n_min", type=int, default=2)
    p.add_argument("--n_max", type=int, default=None, help="default = total speakers")
    p.add_argument("--k", type=int, default=10, help="pseudo speakers per n")

    # GMM controls (like your script), meow
    p.add_argument("--gmm_components", type=int, default=4)
    p.add_argument("--gmm_select_trials", type=int, default=10)
    p.add_argument("--gmm_seed_offset", type=int, default=0)
    p.add_argument("--reg_covar", type=float, default=1e-6)

    # Sampling controls
    p.add_argument("--cov_scale", type=float, default=1.0, help="scale cov by cov_scale^2 when sampling pseudo speakers")

    p.add_argument("--out_csv", type=Path, required=True)
    return p.parse_args()


def fit_pca(Xv: np.ndarray, pca_dim: int):
    mu_x = Xv.mean(axis=0, keepdims=True)  # (1,D)
    Xc = Xv - mu_x
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Vt[:pca_dim]          # (pca_dim, D)
    Z = Xc @ P.T              # (K, pca_dim)
    return mu_x, P, Z


def weight_evenness_score(weights: np.ndarray) -> float:
    """
    Smaller is better. L2 distance to uniform distribution.
    """
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


def min_dist_to_set(x: np.ndarray, Y: np.ndarray) -> float:
    """
    x: (d,), Y: (m,d)
    returns min_j ||x - Y[j]||
    """
    diff = Y - x[None, :]
    d2 = np.sum(diff * diff, axis=1)
    return float(np.sqrt(np.min(d2)))


def main(args):
    assert args.pca_dim >= 1, "Meow: --pca_dim must be >= 1."
    assert args.seed >= 0, "Meow: --seed must be >= 0."
    assert args.k >= 1, "Meow: --k must be >= 1."
    assert args.n_min >= 2, "Meow: --n_min must be >= 2."
    assert args.gmm_components >= 1, "Meow: --gmm_components must be >= 1."
    assert args.gmm_select_trials >= 1, "Meow: --gmm_select_trials must be >= 1."
    assert args.cov_scale > 0, "Meow: --cov_scale must be > 0."
    assert args.reg_covar >= 0, "Meow: --reg_covar must be >= 0."

    rng = np.random.default_rng(args.seed)

    transforms_dir = Path(args.transforms_path)
    transforms_file = transforms_dir / "transforms.npy"
    if not transforms_file.exists():
        raise FileNotFoundError(f"Meow: missing file: {transforms_file}")

    transforms = np.load(transforms_file, allow_pickle=True).item()
    keys = list(transforms.keys())

    # Stack transforms and flatten to vectors
    X = np.stack([np.asarray(transforms[k]) for k in keys], axis=0)  # (K, r, feat)
    K, r_dim, feat_dim = X.shape
    Xv = X.reshape(K, r_dim * feat_dim).astype(np.float64)  # (K, D)

    # One PCA for entire run, meow
    _mu_x, _P, Z_all = fit_pca(Xv, args.pca_dim)  # (K, pca_dim)
    N = Z_all.shape[0]

    n_max = int(args.n_max) if args.n_max is not None else N
    n_max = min(n_max, N)
    if args.n_min > n_max:
        raise ValueError(f"Meow: n_min={args.n_min} > n_max={n_max}")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for n in tqdm(range(args.n_min, n_max + 1), desc="Meow n"):
        # Fit GMM on n sampled real speakers (in PCA space), meow
        fit_idx = rng.choice(N, size=n, replace=False)
        Z_fit = Z_all[fit_idx]  # (n, d)

        gmm, gmm_score, gmm_trial = fit_best_gmm(
            Z_fit,
            n_components=args.gmm_components,
            trials=args.gmm_select_trials,
            seed=args.seed,
            seed_offset=args.gmm_seed_offset,
            reg_covar=args.reg_covar,
        )

        # Sample n-1 real speakers for comparison set
        real_idx = rng.choice(N, size=n - 1, replace=False)
        Z_real = Z_all[real_idx]  # (n-1, d)

        # Sample k pseudo speakers from GMM, compute min distances to real set
        dists = []
        for _ in range(args.k):
            z_pseudo, _comp = sample_from_gmm(rng, gmm, cov_scale=args.cov_scale)
            dists.append(min_dist_to_set(z_pseudo, Z_real))

        mean_min_dist = float(np.mean(dists)) if len(dists) else float("nan")

        rows.append(
            {
                "pca_dim": args.pca_dim,
                "seed": args.seed,
                "n": n,
                "k": args.k,
                "gmm_components": args.gmm_components,
                "gmm_select_trials": args.gmm_select_trials,
                "gmm_selected_trial": gmm_trial,
                "gmm_evenness_l2": float(gmm_score),
                "cov_scale": args.cov_scale,
                "reg_covar": args.reg_covar,
                "mean_min_dist_pseudo_to_real": mean_min_dist,
            }
        )

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "pca_dim",
                "seed",
                "n",
                "k",
                "gmm_components",
                "gmm_select_trials",
                "gmm_selected_trial",
                "gmm_evenness_l2",
                "cov_scale",
                "reg_covar",
                "mean_min_dist_pseudo_to_real",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"Meow: wrote CSV to {out_csv}")


if __name__ == "__main__":
    args = check_argv()
    main(args)