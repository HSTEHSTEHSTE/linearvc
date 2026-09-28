from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import argparse
import numpy as np
import time
import torch
import torchaudio

from linearvc.utils import pca_transform

device = "cuda" if torch.cuda.is_available() else "cpu"


def check_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument("--librispeech_dir", type=Path, required=True)
    parser.add_argument("--subset", type=str, default="train-clean-100")

    parser.add_argument("--pca", type=Path, default=None)
    parser.add_argument("--exclude", type=Path, default=None)

    parser.add_argument("--num_frames", type=int, default=32768)
    parser.add_argument("--rank", type=int, default=100)

    parser.add_argument(
        "--num_index",
        type=int,
        default=-1,
        help="Max number of successful speaker UPDATES (targets); -1 means update with all speakers",
    )

    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("/home/hltcoe/xli/ARTS/linearvc/exp/content_factorization"),
    )

    parser.add_argument("--print_every", type=int, default=1)
    return parser.parse_args()


def load_pca(pca_path: Path | None):
    if pca_path is None:
        return None
    print("Reading PCA:", pca_path)
    p = np.load(pca_path)
    return {key: torch.from_numpy(p[key]).float().to(device) for key in p}


def load_exclude(exclude_path: Path | None):
    if exclude_path is None:
        return set()
    print("Reading exclude list:", exclude_path)
    exclude_utterances = set()
    with open(exclude_path) as f:
        for line in f:
            exclude_utterances.add(line.strip())
    return exclude_utterances


def extract_speaker_features(wavlm, speaker_dir: Path, pca, exclude_utterances: set):
    feats = []
    for wav_fn in sorted(speaker_dir.rglob("*/*.flac")):
        if wav_fn.stem in exclude_utterances:
            continue

        wav, _ = torchaudio.load(str(wav_fn))
        wav = wav.to(device)

        with torch.inference_mode():
            x, _ = wavlm.extract_features(wav, output_layer=6)  # (1, T, 1024)

        if pca is not None:
            x = pca_transform(x, pca["mean"], pca["components"], pca["explained_variance"])

        feats.append(x.squeeze(0).cpu().numpy().astype(np.float32))  # (T, 1024)

    if len(feats) == 0:
        return None
    return np.vstack(feats)


def align(src, refs):
    neighbors = NearestNeighbors(n_neighbors=1, metric="cosine")
    neighbors.fit(refs)
    _, indices = neighbors.kneighbors(src)
    return refs[indices.squeeze(), :]


def svd_low_rank_update(U, S, VT, P, Q, eps=1e-12):
    m, k = U.shape
    k2, n = VT.shape
    assert k == k2
    r = P.shape[1]
    assert Q.shape == (n, r)

    P = P.astype(np.float64, copy=False)
    Q = Q.astype(np.float64, copy=False)

    P_proj = U.T @ P
    Q_proj = VT @ Q

    P_perp = P - U @ P_proj
    Q_perp = Q - (VT.T @ Q_proj)

    if np.linalg.norm(P_perp) > eps:
        Up, Rp = np.linalg.qr(P_perp, mode="reduced")
    else:
        Up = np.zeros((m, 0), dtype=np.float64)
        Rp = np.zeros((0, r), dtype=np.float64)

    if np.linalg.norm(Q_perp) > eps:
        Vp, Rq = np.linalg.qr(Q_perp, mode="reduced")
    else:
        Vp = np.zeros((n, 0), dtype=np.float64)
        Rq = np.zeros((0, r), dtype=np.float64)

    rp = Up.shape[1]
    rq = Vp.shape[1]

    K = np.zeros((k + rp, k + rq), dtype=np.float64)
    K[:k, :k] = np.diag(S)

    L = np.vstack([P_proj, Rp])
    R = np.vstack([Q_proj, Rq])
    K = K + L @ R.T

    Uk, Sk, VTk = np.linalg.svd(K, full_matrices=False)

    U_aug = np.column_stack([U, Up])
    V_aug = np.column_stack([VT.T, Vp])

    U_new = U_aug @ Uk
    VT_new = (V_aug @ VTk.T).T
    S_new = Sk
    return U_new, S_new, VT_new


def truncate_svd(U, S, VT, max_rank: int):
    keep = min(max_rank, len(S))
    return U[:, :keep], S[:keep], VT[:keep, :]


def main(args):
    subset_dir = args.librispeech_dir / args.subset
    if not subset_dir.is_dir():
        raise FileNotFoundError(f"Subset dir not found: {subset_dir}")

    print("Device:", device)
    print("Subset folder:", subset_dir)
    print("num_frames (source only):", args.num_frames)
    print("final rank:", args.rank)
    print("max updates (num_index):", args.num_index)

    wavlm = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, device=device)
    pca = load_pca(args.pca)
    exclude_utterances = load_exclude(args.exclude)

    speaker_dirs = sorted([p for p in subset_dir.glob("*") if p.is_dir()])
    print("Num speaker dirs:", len(speaker_dirs))
    if len(speaker_dirs) == 0:
        raise RuntimeError("No speaker dirs found.")

    block_r = 1024

    # Single source speaker: first one
    src_spk_dir = speaker_dirs[0]
    src_spk = src_spk_dir.stem

    # Init speaker == source in this setup
    init_spk_dir = speaker_dirs[0]
    init_spk = init_spk_dir.stem

    out_path = (
        args.output_root
        / ("librispeech_" + args.subset)
        / ("rank_" + str(args.rank))
        / ("src_" + src_spk)
    )
    out_path.mkdir(parents=True, exist_ok=True)

    xs_dir = out_path / "XS"
    xs_dir.mkdir(parents=True, exist_ok=True)

    print("\n==============================")
    print("Single source speaker:", src_spk)
    print("Writing to:", out_path)
    print("Writing aligned blocks to:", xs_dir)

    # Source extraction (truncate only source)
    t_src = time.time()
    X_src_full = extract_speaker_features(wavlm, src_spk_dir, pca, exclude_utterances)
    t_src = time.time() - t_src
    if X_src_full is None or X_src_full.shape[0] < args.num_frames:
        raise RuntimeError(f"Source {src_spk} has insufficient frames (extract {t_src:.2f}s)")

    X_src = X_src_full[: args.num_frames]
    print(f"Source extracted: frames_total={X_src_full.shape[0]} using={X_src.shape[0]} (time {t_src:.2f}s)")

    # Init block: init==source, so aligned block is X_src
    X0 = X_src.astype(np.float32)  # (num_frames, 1024)
    np.save(xs_dir / f"{init_spk}.npy", X0)

    t_svd0 = time.time()
    U, S, VT = np.linalg.svd(X0.astype(np.float64), full_matrices=False)
    t_svd0 = time.time() - t_svd0

    U, S, VT = truncate_svd(U, S, VT, 1024)
    used_speakers = [init_spk]
    print(f"Init full SVD time {t_svd0:.2f}s; svd_rank_now={len(S)}")

    successful_updates = 0

    for spk_dir in tqdm(speaker_dirs[1:], desc="Updating speakers"):
        if args.num_index != -1 and successful_updates >= args.num_index:
            break

        spk = spk_dir.stem

        t_ex = time.time()
        X_spk = extract_speaker_features(wavlm, spk_dir, pca, exclude_utterances)
        t_ex = time.time() - t_ex
        if X_spk is None:
            continue

        t_al = time.time()
        Xj = align(X_src, X_spk).astype(np.float32)  # (num_frames, 1024)
        t_al = time.time() - t_al

        # Save this aligned block as XS/<speaker>.npy
        np.save(xs_dir / f"{spk}.npy", Xj)

        # Expand width by 1024 zeros then apply rank-1024 update
        VT = np.hstack([VT, np.zeros((VT.shape[0], block_r), dtype=np.float64)])
        n = VT.shape[1]
        Q = np.zeros((n, block_r), dtype=np.float64)
        Q[n - block_r : n, :] = np.eye(block_r, dtype=np.float64)

        t_up = time.time()
        U, S, VT = svd_low_rank_update(U, S, VT, Xj.astype(np.float64), Q)
        t_up = time.time() - t_up

        t_tr = time.time()
        U, S, VT = truncate_svd(U, S, VT, 1024)
        t_tr = time.time() - t_tr

        used_speakers.append(spk)
        successful_updates += 1

        if args.print_every > 0 and (successful_updates % args.print_every == 0):
            print(
                f"[update {successful_updates}"
                + (f"/{args.num_index}]" if args.num_index != -1 else "]")
                + f" {spk}: spk_frames={X_spk.shape[0]} src_frames={X_src.shape[0]} "
                  f"extract={t_ex:.2f}s align={t_al:.2f}s update={t_up:.2f}s trunc={t_tr:.3f}s "
                  f"svd_rank_now={len(S)} width_now={VT.shape[1]}"
            )

    # Final truncation to requested rank
    r = min(args.rank, len(S))
    U_r = U[:, :r].astype(np.float32)
    S_r = S[:r].astype(np.float32)
    VT_r = VT[:r, :].astype(np.float32)

    expected_cols = len(used_speakers) * block_r
    if VT_r.shape[1] != expected_cols:
        raise ValueError(f"Expected {expected_cols} cols, got {VT_r.shape[1]}")

    VT_rs = VT_r.reshape(r, len(used_speakers), block_r).swapaxes(0, 1)
    transforms = {spk: VT_rs[i, :, :] for i, spk in enumerate(used_speakers)}

    np.save(out_path / "U.npy", U_r)
    np.save(out_path / "S.npy", S_r)
    np.save(out_path / "VT.npy", VT_rs)
    np.save(out_path / "transforms.npy", transforms)
    np.save(out_path / "used_speakers.npy", np.array(used_speakers, dtype=object))

    print("Wrote:", out_path)


if __name__ == "__main__":
    args = check_argv()
    main(args)