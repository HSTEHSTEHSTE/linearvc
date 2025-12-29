import argparse, random
import torch, torchaudio
import numpy as np
from pathlib import Path
from tqdm import tqdm
from linearvc import linearvc

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def check_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_dir",
        type=Path,
        help="source speech directory",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        help="output speech directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
    )
    parser.add_argument(
        "--content_factorization_path",
        type=Path,
        help="root path to content factorization"
    )
    return parser.parse_args()

def get_spk_mapping(spks, seed):
    random.seed(seed)
    if len(spks) < 2:
        raise ValueError("Not enough speakers")

    while True:
        shuffled = spks[:]
        random.shuffle(shuffled)

        # check no one maps to themselves
        if all(a != b for a, b in zip(spks, shuffled)):
            return dict(zip(spks, shuffled))

def main(args):
    print("Source dir: ", args.src_dir)
    print("Out dir: ", args.out_dir)
    print("Content factorization path: ", args.content_factorization_path)
    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)

    content_path = Path(args.content_factorization_path)
    transforms = np.load(content_path / 'transforms.npy', allow_pickle=True).item()

    extensions = ['wav', 'flac']
    spks_long = src_dir.iterdir()
    spks = []
    for spk_long in spks_long:
        spks.append(str(spk_long).split('/')[-1])
    spk_map = get_spk_mapping(spks, args.seed)

    # Load all the required models
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

    for spk_src in tqdm(spk_map, total=len(spks)):
        spk_tgt = spk_map[spk_src]

        wavs_src = []
        for extension in extensions:
            wavs_src += (src_dir / spk_src).rglob('*.' + extension)

        for wav_src in wavs_src:
            input_features = linearvc_model.get_features(wav_src)
            input_features = input_features.cpu().numpy()
            out_features = torch.tensor(np.dot(np.dot(input_features, np.linalg.pinv(transforms[spk_src])), transforms[spk_tgt])).to(device)
            wav_hat = hifigan(out_features.unsqueeze(0)).squeeze(0).detach().cpu()
            (out_dir / spk_src).mkdir(parents=True, exist_ok=True)
            torchaudio.save(out_dir / spk_src / (wav_src.stem + '.wav'), wav_hat, 16000)


if __name__ == "__main__":
    args = check_argv()
    main(args)