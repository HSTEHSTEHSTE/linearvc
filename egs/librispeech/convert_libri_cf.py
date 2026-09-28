import argparse, random
import torch, torchaudio
import numpy as np
from pathlib import Path
from tqdm import tqdm
from linearvc import linearvc

device = "cuda" if torch.cuda.is_available() else "cpu"


def check_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=Path, help="source speech directory")
    parser.add_argument("--out_dir", type=Path, help="output speech directory")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--content_factorization_path", type=Path, help="root path to content factorization")
    parser.add_argument(
        "--pinv_type",
        type=str,
        default="UTXSS",
        help="Pinv matrix. src, anchor, ST, or UTXSS",
    )
    return parser.parse_args()


def get_src_spks(src_dir: Path):
    spks = []
    for spk_dir in src_dir.iterdir():
        if spk_dir.is_dir():
            spks.append(spk_dir.name)
    return sorted(spks)


def get_spk_mapping_from_targets(src_spks, tgt_spks, seed):
    """
    Map each source speaker to a random target speaker drawn from tgt_spks.
    If a source speaker name exists in tgt_spks and there is >1 target,
    avoid mapping to itself.
    """
    random.seed(seed)
    if len(tgt_spks) == 0:
        raise ValueError("No target speakers available in transforms.")
    if len(tgt_spks) == 1 and src_spks and src_spks[0] in tgt_spks:
        # still allowed; will map to the only target (itself)
        pass

    mapping = {}
    for s in src_spks:
        candidates = tgt_spks
        if s in tgt_spks and len(tgt_spks) > 1:
            candidates = [t for t in tgt_spks if t != s]
        mapping[s] = random.choice(candidates)
    return mapping


def main(args):
    print("Source dir:", args.src_dir)
    print("Out dir:", args.out_dir)
    print("Content factorization path:", args.content_factorization_path)
    print("Pinv type:", args.pinv_type)

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)

    content_path = Path(args.content_factorization_path)
    spk_anchor = content_path.name.split("_")[-1]

    transforms = np.load(content_path / "transforms.npy", allow_pickle=True).item()
    tgt_spks = sorted(list(transforms.keys()))  # <-- target set from transforms keys

    if args.pinv_type == "ST":
        ST = np.load(content_path / "ST.npy")
    elif args.pinv_type == "UTXSS":
        ST = np.load(content_path / "UTXSS.npy")
    elif args.pinv_type == "anchor":
        ST = np.linalg.pinv(transforms[spk_anchor])

    # Source speakers come from src_dir; targets come from transforms keys
    src_spks = get_src_spks(src_dir)
    spk_map = get_spk_mapping_from_targets(src_spks, tgt_spks, args.seed)

    # Load models
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
    linearvc_model = linearvc.LinearVC(wavlm, hifigan, device)

    extensions = ["wav", "flac"]

    for spk_src in tqdm(spk_map, total=len(src_spks)):
        spk_tgt = spk_map[spk_src]

        wavs_src = []
        for extension in extensions:
            wavs_src += (src_dir / spk_src).rglob("*." + extension)

        for wav_src in wavs_src:
            input_features = linearvc_model.get_features(wav_src).cpu().numpy()

            if args.pinv_type == "src":
                # Note: requires transforms[spk_src] to exist; if src speaker not in transforms this will error.
                out_features = torch.tensor(
                    input_features @ np.linalg.pinv(transforms[spk_src]) @ transforms[spk_tgt]
                ).to(device).float()
            else:
                out_features = torch.tensor(
                    input_features @ ST @ transforms[spk_tgt]
                ).to(device).float()

            wav_hat = hifigan(out_features.unsqueeze(0)).squeeze(0).detach().cpu()
            (out_dir / spk_src).mkdir(parents=True, exist_ok=True)
            torchaudio.save(str(out_dir / spk_src / (wav_src.stem + ".wav")), wav_hat, 16000)


if __name__ == "__main__":
    args = check_argv()
    main(args)