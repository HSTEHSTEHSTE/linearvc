import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

GT_ROOT_DEFAULT = Path("/home/hltcoe/xli/ARTS/linearvc/exp/interspeech/cf/spk/orig")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--converted_embed_dir", type=Path, required=True)
    ap.add_argument("--gt_embed_dir", type=Path, default=GT_ROOT_DEFAULT)
    ap.add_argument("--min_gt_per_spk", type=int, default=1)
    return ap.parse_args()

def iter_npy_files(root: Path):
    for p in root.rglob("*.npy"):
        if p.is_file():
            yield p

def speaker_from_path(root: Path, p: Path):
    rel = p.relative_to(root)
    return rel.parts[0]  # root/<spk>/...

def load_embs_by_speaker(root: Path):
    by_spk = {}
    for p in iter_npy_files(root):
        spk = speaker_from_path(root, p)
        e = np.asarray(np.load(p)).reshape(-1)
        by_spk.setdefault(spk, []).append(e)
    for spk in list(by_spk.keys()):
        by_spk[spk] = np.stack(by_spk[spk], axis=0)  # (N,D)
    return by_spk

def main():
    args = parse_args()

    gt_by_spk = load_embs_by_speaker(args.gt_embed_dir)
    gt_by_spk = {s: X for s, X in gt_by_spk.items() if X.shape[0] >= args.min_gt_per_spk}
    spks = sorted(gt_by_spk.keys())
    if not spks:
        raise RuntimeError("Meow: no GT speakers found.")

    D = next(iter(gt_by_spk.values())).shape[1]

    # speaker vectors: mean of GT embeddings
    gt_vec = {s: gt_by_spk[s].mean(axis=0) for s in spks}   # (D,)
    gt_vec_mat = np.stack([gt_vec[s] for s in spks], axis=0)  # (S,D)

    # 1) GT intra-speaker: mean ||x - mu_s||
    intra = {}
    for s in spks:
        X = gt_by_spk[s]
        mu = gt_vec[s][None, :]
        intra[s] = float(np.linalg.norm(X - mu, axis=-1).mean())

    # 2) GT to other speaker vectors:
    #    (2a) mean distance to all other vectors
    #    (2b) mean of min distance to any other vector (exclude self)
    inter_mean = {}
    inter_min = {}
    for i, s in enumerate(spks):
        X = gt_by_spk[s]                          # (N,D)
        others = np.delete(gt_vec_mat, i, axis=0) # (S-1,D)

        d = np.linalg.norm(X[:, None, :] - others[None, :, :], axis=-1)  # (N,S-1)
        inter_mean[s] = float(d.mean())
        inter_min[s] = float(d.min(axis=1).mean())  # per-embedding nearest other speaker vector, averaged

    # 3) Converted: average of MIN distance to any GT speaker vector
    conv_files = sorted(list(iter_npy_files(args.converted_embed_dir)))
    if len(conv_files) == 0:
        raise RuntimeError("Meow: no converted embeddings found.")

    min_dists_all = []
    min_dists_by_conv_spk = {}

    for p in tqdm(conv_files, desc="Meow converted->GT (min)"):
        e = np.asarray(np.load(p)).reshape(-1)
        if e.shape[0] != D:
            raise RuntimeError(f"Meow: embedding dim mismatch at {p} (got {e.shape[0]}, expected {D})")

        d = np.linalg.norm(gt_vec_mat - e[None, :], axis=-1)  # (S,)
        dmin = float(d.min())
        min_dists_all.append(dmin)

        conv_spk = speaker_from_path(args.converted_embed_dir, p)
        min_dists_by_conv_spk.setdefault(conv_spk, []).append(dmin)

    conv_min_mean = float(np.mean(min_dists_all))

    print("Meow results (Euclidean / L2):")
    print(f"GT intra-speaker (avg over speakers): {float(np.mean(list(intra.values()))):.6f}")
    print(f"GT->others mean (avg over speakers): {float(np.mean(list(inter_mean.values()))):.6f}")
    print(f"GT->others min  (avg over speakers): {float(np.mean(list(inter_min.values()))):.6f}")
    print(f"Converted->GT min (avg over converted files): {conv_min_mean:.6f}")

    print("\nMeow per-GT-speaker:")
    for s in spks:
        print(
            f"{s}\tintra={intra[s]:.6f}"
            f"\tto_others_mean={inter_mean[s]:.6f}"
            f"\tto_others_min={inter_min[s]:.6f}"
            f"\tN_gt={gt_by_spk[s].shape[0]}"
        )

    print("\nMeow per-converted-speaker (min-to-GT):")
    for s in sorted(min_dists_by_conv_spk.keys()):
        print(f"{s}\tmin_to_gt={float(np.mean(min_dists_by_conv_spk[s])):.6f}\tN_conv={len(min_dists_by_conv_spk[s])}")

if __name__ == "__main__":
    main()