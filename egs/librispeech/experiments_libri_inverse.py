import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np


def check_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transform_root_dir",
        type=Path,
        required=True,
        help="transform root directory",
    )
    parser.add_argument(
        "--transform_type",
        type=str,
        required=True,
        help="Type of transform. ST, UTXSS, or USTXS",
        choices=["ST", "UTXSS", "USTXS"],
    )
    parser.add_argument(
        "--xs_folder",
        action="store_true",
        help="If set, load aligned blocks from transform_root_dir/XS/*.npy instead of transform_root_dir/XS.npy",
    )
    return parser.parse_args()


def load_XS(transform_root_dir: Path, use_xs_folder: bool):
    """
    Returns XS with shape (t, k*d) as expected by downstream code.
    If using folder: expects transform_root_dir/XS/<speaker_id>.npy each (t, d=1024).
    """
    if not use_xs_folder:
        return np.load(transform_root_dir / "XS.npy")

    xs_dir = transform_root_dir / "XS"
    if not xs_dir.is_dir():
        raise FileNotFoundError(f"Expected XS folder at: {xs_dir}")

    xs_files = sorted(xs_dir.glob("*.npy"))
    if len(xs_files) == 0:
        raise FileNotFoundError(f"No .npy files found in: {xs_dir}")

    blocks = []
    t_expected = None
    d_expected = None

    for fn in tqdm(xs_files):
        x = np.load(fn)  # expected (t, 1024)
        if x.ndim != 2:
            raise ValueError(f"{fn} expected 2D array (t,d), got shape {x.shape}")
        if t_expected is None:
            t_expected = x.shape[0]
            d_expected = x.shape[1]
        else:
            if x.shape[0] != t_expected or x.shape[1] != d_expected:
                raise ValueError(
                    f"Mismatched XS block shapes: first (t,d)=({t_expected},{d_expected}), "
                    f"but {fn} has {x.shape}"
                )
        blocks.append(x)

    # Concatenate along feature axis -> (t, k*d)
    XS = np.concatenate(blocks, axis=1)
    return XS


def main(args):
    transform_root_dir = Path(args.transform_root_dir)

    if args.transform_type == "ST":
        # compute ST
        VT = np.load(transform_root_dir / "VT.npy")  # [k, r, d]
        k, r, d = VT.shape
        VT = VT.reshape([-1, VT.shape[-1]])  # [k * r, d]
        target = np.identity(r)  # [r, r]
        target = np.expand_dims(target, 0)  # [1, r, r]
        target = target.repeat(k, axis=0)  # [k, r, r]
        target = target.reshape([-1, target.shape[-1]])  # [k * r, r]
        lstsq_solution = np.linalg.lstsq(VT, target, rcond=None)
        ST = lstsq_solution[0]  # [d, r]
        np.save(transform_root_dir / "ST.npy", ST)

    else:
        # compute UTXSS or USTXS
        print("Loading XS")
        XS = load_XS(transform_root_dir, args.xs_folder)  # [t, k * d]
        XS = XS.reshape([XS.shape[0], -1, 1024])  # [t, k, d]
        t, k, d = XS.shape
        XS = XS.reshape([-1, d])  # [t*k, d]

        U = np.load(transform_root_dir / "U.npy")  # [t, r]
        S = np.load(transform_root_dir / "S.npy")  # [r]
        S = np.diag(S)  # [r, r]
        US = np.matmul(U, S)  # [t, r]
        US = np.expand_dims(US, 1).repeat(k, 1)  # [t, k, r]
        r = US.shape[-1]
        US = US.reshape([-1, r])  # [t*k, r]

        U_rep = np.expand_dims(U, 1).repeat(k, 1)  # [t, k, r]
        U_rep = U_rep.reshape([-1, r])  # [t*k, r]

        print("Solving least squares")
        if args.transform_type == "UTXSS":
            lstsq_solution = np.linalg.lstsq(XS, U_rep, rcond=None)
            UTXS = lstsq_solution[0]
            UTXSS = np.matmul(UTXS, S)
            np.save(transform_root_dir / "UTXSS.npy", UTXSS)
        else:
            US_scaled = US / 100
            lstsq_solution = np.linalg.lstsq(XS, US_scaled, rcond=None)
            np.save(transform_root_dir / "USTXS.npy", lstsq_solution[0] * 100)


if __name__ == "__main__":
    args = check_argv()
    main(args)