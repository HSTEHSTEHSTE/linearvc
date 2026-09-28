import argparse, json
import torch, torchaudio
import numpy as np
from pathlib import Path
from tqdm import tqdm
from linearvc.linearvc import LinearVC
from sklearn.mixture import GaussianMixture

device = "cuda" if torch.cuda.is_available() else "cpu"

def check_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument("--librispeech_root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True, help="output speech directory")
    parser.add_argument("--num_utt_per_speaker", type=int, default=20)
    parser.add_argument("--set_num", type=int, required=True)
    parser.add_argument("--feat_path", type=Path)  # unused now, kept for CLI compatibility
    parser.add_argument("--frame_limit", type=int, default=500)  # unused now, kept for CLI compatibility
    parser.add_argument("--hifigan_path", type=Path, default=None)
    parser.add_argument("--set", type=str, default="librispeech", help="must be librispeech now")
    parser.add_argument("--pca_dim", type=int, default=10)

    parser.add_argument("--gmm_components", type=int, default=4)
    parser.add_argument("--cov_scale", type=float, default=1.5)

    parser.add_argument("--gmm_select_trials", type=int, default=10, help="fit GMM this many times and pick most even weights")
    parser.add_argument("--gmm_seed_offset", type=int, default=0, help="added to seed for each trial")

    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()

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

def fit_best_gmm(Z: np.ndarray, *, n_components: int, trials: int, seed: int, seed_offset: int):
    best_gmm = None
    best_score = float("inf")
    best_trial = None

    for t in range(trials):
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            reg_covar=1e-6,
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

def main(args):
    assert args.set == "librispeech", "Meow: this edited script supports only --set librispeech."
    assert args.gmm_components >= 1, "Meow: --gmm_components must be >= 1."
    assert args.cov_scale > 0, "Meow: --cov_scale must be > 0."
    assert args.gmm_select_trials >= 1, "Meow: --gmm_select_trials must be >= 1."

    librispeech_root = Path(args.librispeech_root)
    out_dir = Path(args.out_dir) / str(args.set_num)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    with open("linearvc/egs/librispeech/libri_test/speakers.json", "r") as file:
        speakers = json.load(file)
    in_speakers = speakers["lists"]["test-clean_source"]

    wavlm = torch.hub.load(
        "bshall/knn-vc",
        "wavlm_large",
        trust_repo=True,
        progress=True,
        device=device,
    )

    if args.hifigan_path is None:
        hifigan, _ = torch.hub.load(
            "bshall/knn-vc",
            "hifigan_wavlm",
            trust_repo=True,
            prematched=True,
            progress=True,
            device=device,
        )
    else:
        import os
        from linearvc.hifigan.models import Generator

        class AttrDict(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.__dict__ = self

        def load_hifigan_checkpoint(filepath, device):
            checkpoint_dict = torch.load(filepath, map_location=device)
            return checkpoint_dict

        config_file = os.path.join(os.path.split(args.hifigan_path)[0], "config.json")
        with open(config_file) as f:
            json_config = json.loads(f.read())
        h = AttrDict(json_config)
        hifigan = Generator(h).to(device)
        state_dict_g = load_hifigan_checkpoint(args.hifigan_path, device)
        hifigan.load_state_dict(state_dict_g["generator"])

    linearvc_model = LinearVC(wavlm, hifigan, device)

    transforms_path = Path("linearvc/exp/interspeech/cf/transforms")
    ranks = [75]

    for rank in tqdm(ranks, desc="Meow ranks"):
        transforms = np.load(
            transforms_path / str(args.set_num) / f"rank_{rank}" / "transforms.npy",
            allow_pickle=True
        ).item()

        W2_np = np.load(transforms_path / str(args.set_num) / f"rank_{rank}" / "ST.npy")
        W2 = torch.tensor(W2_np).to(device).float()

        keys = list(transforms.keys())
        X = np.stack([np.asarray(transforms[k]) for k in keys], axis=0)  # (K, r, 1024)
        K, r_dim, feat_dim = X.shape
        D = r_dim * feat_dim
        Xv = X.reshape(K, D).astype(np.float64)

        mu_x, P, Z = fit_pca(Xv, args.pca_dim)

        gmm, gmm_score, gmm_trial = fit_best_gmm(
            Z,
            n_components=args.gmm_components,
            trials=args.gmm_select_trials,
            seed=args.seed,
            seed_offset=args.gmm_seed_offset,
        )
        print(f"Meow: selected GMM trial={gmm_trial} weights={gmm.weights_} evenness(L2)={gmm_score:.6g}")

        # Manual sampler with optional covariance scaling
        def sample_gmm_target_transform():
            comp = int(rng.choice(args.gmm_components, p=gmm.weights_))
            cov = gmm.covariances_[comp] * (args.cov_scale ** 2)
            z = rng.multivariate_normal(mean=gmm.means_[comp], cov=cov)
            x = mu_x[0] + (z @ P)
            T = x.reshape(r_dim, feat_dim).astype(X.dtype, copy=False)
            return T, z, comp

        tgt_by_spk = {}
        for spk in in_speakers:
            T_tgt_np, z, comp = sample_gmm_target_transform()
            T_tgt = torch.tensor(T_tgt_np).to(device).float()
            tgt_by_spk[spk] = (T_tgt, z, comp)

            spk_meta_dir = out_dir / "W2_GMM_PCA" / f"gmm{args.gmm_components}_covs{args.cov_scale:g}" / str(rank) / spk
            spk_meta_dir.mkdir(parents=True, exist_ok=True)
            np.save(spk_meta_dir / "gmm_target_T.npy", T_tgt_np)
            np.save(spk_meta_dir / "gmm_target_z.npy", z)
            (spk_meta_dir / "gmm_target_component.txt").write_text(str(comp))
            (spk_meta_dir / "gmm_target_meta.txt").write_text(
                "method=gmm_pca\n"
                f"pca_dim={args.pca_dim}\n"
                f"gmm_components={args.gmm_components}\n"
                f"cov_scale={args.cov_scale}\n"
                f"gmm_select_trials={args.gmm_select_trials}\n"
                f"gmm_selected_trial={gmm_trial}\n"
                f"gmm_evenness_l2={gmm_score}\n"
                f"rank={rank}\n"
                f"seed={args.seed}\n"
            )

        for in_speaker in tqdm(in_speakers, desc=f"Meow speakers (rank={rank})", leave=False):
            spk_wavs = sorted((librispeech_root / "test-clean" / in_speaker).rglob("*.flac"))
            if len(spk_wavs) == 0:
                continue

            T_tgt, _z, _comp = tgt_by_spk[in_speaker]

            out_spk_dir = out_dir / "W2_GMM_PCA" / f"gmm{args.gmm_components}_covs{args.cov_scale:g}" / str(rank) / in_speaker
            out_spk_dir.mkdir(parents=True, exist_ok=True)

            for spk_wav in spk_wavs[: args.num_utt_per_speaker]:
                with torch.no_grad():
                    input_features = linearvc_model.get_features(str(spk_wav))
                    out_features = torch.matmul(torch.matmul(input_features, W2), T_tgt).float()
                    wav_hat = hifigan(out_features.unsqueeze(0)).squeeze(0).detach().cpu()

                torchaudio.save(
                    str(out_spk_dir / f"{spk_wav.stem}.wav"),
                    wav_hat,
                    16000
                )

if __name__ == "__main__":
    args = check_argv()
    main(args)