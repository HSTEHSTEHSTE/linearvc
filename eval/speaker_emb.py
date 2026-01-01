import argparse
import numpy as np
import torch
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
from tqdm import tqdm

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def check_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--converted_dir",
        type=Path,
        help="converted speech directory",
    )
    parser.add_argument(
        "--out_embed_dir",
        type=Path,
        help="directory to store Resemblyzer output",
    )
    return parser.parse_args()

def main(args):
    converted_dir = Path(args.converted_dir)
    out_embed_dir = Path(args.out_embed_dir)
    out_embed_dir.mkdir(parents=True, exist_ok=True)

    extensions = ['wav', 'flac']
    spks_long = converted_dir.iterdir()
    spks = []
    for spk_long in spks_long:
        spks.append(str(spk_long).split('/')[-1])

    encoder = VoiceEncoder()

    for spk in tqdm(spks):
        wavs = []
        embeds = []
        for extension in extensions:
            wavs += (converted_dir / spk).rglob('*.' + extension)

        for wav in tqdm(wavs):
            wav = preprocess_wav(wav)
            embed = encoder.embed_utterance(wav)
            embeds.append(embed)

        embeds_avg = np.array(embeds).mean(axis=0)

        np.save(out_embed_dir / (spk + '.npy'), embeds_avg)


if __name__ == "__main__":
    args = check_argv()
    main(args)