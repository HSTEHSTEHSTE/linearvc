import argparse
import csv
from math import comb
from pathlib import Path

import numpy as np
from tqdm import tqdm


def check_argv():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--transforms_path",
        type=Path,
        required=True,
        help="directory that directly contains transforms.npy",
    )
    parser.add_argument(
        "--pca_dim",
        type=int,
        default=10,
        help="PCA output dimension",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="number of repeated random speaker-subset experiments per n",
    )
    parser.add_argument(
        "--out_csv",
        type=Path,
        required=True,
        help="output CSV path",
    )
    return parser.parse_args()


def fit_pca(Xv: np.ndarray, pca_dim: int):
    mu_x = Xv.mean(axis=0, keepdims=True)  # (1,D)
    Xc = Xv - mu_x
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Vt[:pca_dim]          # (pca_dim, D)
    Z = Xc @ P.T              # (K, pca_dim)
    return mu_x, P, Z


def avg_min_neighbor_distance(Zn: np.ndarray) -> float:
    """
    For each row i, compute min_{j!=i} ||Zn[i]-Zn[j]||, then average over i.
    Zn: (n, pca_dim)
    """
    n = Zn.shape[0]
    if n < 2:
        return float("nan")

    mins = np.empty(n, dtype=np.float64)
    for i in range(n):
        best = float("inf")
        for j in range(n):
            if i == j:
                continue
            d = float(np.linalg.norm(Zn[i] - Zn[j], ord=2))
            if d < best:
                best = d
        mins[i] = best

    return float(mins.mean())


def main(args):
    assert args.pca_dim >= 1, "Meow: --pca_dim must be >= 1."
    assert args.k >= 1, "Meow: --k must be >= 1."

    rng = np.random.default_rng(args.seed)

    transforms_dir = Path(args.transforms_path)
    transforms_file = transforms_dir / "transforms.npy"
    assert transforms_file.exists(), f"Meow: missing file: {transforms_file}"

    transforms = np.load(transforms_file, allow_pickle=True).item()
    keys = list(transforms.keys())  # speaker/ids

    # X: (K, r, feat_dim)
    X = np.stack([np.asarray(transforms[k]) for k in keys], axis=0)
    K, r_dim, feat_dim = X.shape
    D = r_dim * feat_dim

    # Xv: (K, D)
    Xv = X.reshape(K, D).astype(np.float64)

    # Fit PCA ONCE for entire run, meow
    _mu_x, _P, Z = fit_pca(Xv, args.pca_dim)  # Z: (K, pca_dim)
    Nspk = Z.shape[0]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for n in tqdm(range(2, Nspk + 1), desc="Meow n"):
        # Repeat k times; unless impossible to get k distinct experiments
        max_unique = None
        try:
            max_unique = comb(Nspk, n)
        except Exception:
            max_unique = None

        target_trials = int(args.k)
        if max_unique is not None:
            target_trials = min(target_trials, max_unique)

        # Enforce uniqueness only when combinatorics are manageable, meow
        enforce_unique = (max_unique is not None and max_unique <= 5_000_000)

        seen = set()
        vals = []

        attempts = 0
        max_attempts = target_trials * 100  # safety

        while len(vals) < target_trials and attempts < max_attempts:
            idx = tuple(sorted(rng.choice(Nspk, size=n, replace=False).tolist()))
            attempts += 1

            if enforce_unique:
                if idx in seen:
                    continue
                seen.add(idx)

            Zn = Z[np.array(idx)]
            vals.append(avg_min_neighbor_distance(Zn))

        mean_val = float(np.mean(vals)) if len(vals) else float("nan")

        rows.append(
            {
                "pca_dim": args.pca_dim,
                "seed": args.seed,
                "n": n,
                "trials_target": target_trials,
                "trials_done": len(vals),
                "mean_avg_min_neighbor_euclidean": mean_val,
            }
        )

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "pca_dim",
                "seed",
                "n",
                "trials_target",
                "trials_done",
                "mean_avg_min_neighbor_euclidean",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"Meow: wrote CSV to {out_csv}")


if __name__ == "__main__":
    args = check_argv()
    main(args)