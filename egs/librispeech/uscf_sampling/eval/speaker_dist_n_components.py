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

    # pseudo-speaker sampling
    p.add_argument("--k", type=int, default=50, help="pseudo speakers to sample per #components")

    # GMM selection controls (like your script), meow
    p.add_argument("--gmm_select_trials", type=int, default=1)
    p.add_argument("--gmm_seed_offset", type=int, default=0)
    p.add_argument("--reg_covar", type=float, default=1e-6)

    # sampling controls
    p.add_argument("--cov_scale", type=float, default=1.5, help="scale cov by cov_scale^2 when sampling")

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
    diff = Y - x[None, :]
    d2 = np.sum(diff * diff, axis=1)
    return float(np.sqrt(np.min(d2)))


def main(args):
    assert args.pca_dim >= 1, "Meow: --pca_dim must be >= 1."
    assert args.k >= 1, "Meow: --k must be >= 1."
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

    X = np.stack([np.asarray(transforms[k]) for k in keys], axis=0)  # (N, r, feat)
    N, r_dim, feat_dim = X.shape
    Xv = X.reshape(N, r_dim * feat_dim).astype(np.float64)

    # PCA over ALL speakers (n fixed = number of speakers), meow
    _mu_x, _P, Z_all = fit_pca(Xv, args.pca_dim)  # (N, pca_dim)

    # Real speaker reference set: all real speakers
    Z_real = Z_all

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    # sweep components from 2..N (cannot exceed samples)
    for m in tqdm(range(2, N + 1), desc="Meow gmm_components"):
        gmm, gmm_score, gmm_trial = fit_best_gmm(
            Z_all,
            n_components=m,
            trials=args.gmm_select_trials,
            seed=args.seed,
            seed_offset=args.gmm_seed_offset,
            reg_covar=args.reg_covar,
        )

        dists = []
        for _ in range(args.k):
            z_pseudo, _comp = sample_from_gmm(rng, gmm, cov_scale=args.cov_scale)
            dists.append(min_dist_to_set(z_pseudo, Z_real))

        rows.append(
            {
                "num_speakers": N,
                "pca_dim": args.pca_dim,
                "seed": args.seed,
                "gmm_components": m,
                "k": args.k,
                "gmm_select_trials": args.gmm_select_trials,
                "gmm_selected_trial": gmm_trial,
                "gmm_evenness_l2": float(gmm_score),
                "cov_scale": args.cov_scale,
                "reg_covar": args.reg_covar,
                "mean_min_dist_pseudo_to_real": float(np.mean(dists)),
            }
        )

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "num_speakers",
                "pca_dim",
                "seed",
                "gmm_components",
                "k",
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