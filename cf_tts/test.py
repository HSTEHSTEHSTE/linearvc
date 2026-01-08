import argparse
from pathlib import Path
import yaml
import numpy as np
import torch
import torchaudio
import sentencepiece as spm

from linearvc import linearvc
from linearvc.cf_tts.models.tts import ZipVoice

from linearvc.cf_tts.utils.common import invert_normalized_input

# -------------------------
# helpers
# -------------------------

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return ckpt["step"]


# -------------------------
# main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sampling_steps", default=16, type=int)
    parser.add_argument("--target_speaker", default='1272')
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # load models
    # -------------------------

    print("Loading ZipVoice...")
    model = ZipVoice(**cfg["model"]["tts"]["zipvoice"]).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    print("Loading SentencePiece...")
    sp = spm.SentencePieceProcessor()
    sp.load(cfg["training"]["spm_file"])

    print("Loading WavLM + HiFiGAN...")
    wavlm = torch.hub.load(
        "bshall/knn-vc",
        "wavlm_large",
        trust_repo=True,
        device=device
    )
    hifigan, _ = torch.hub.load(
        "bshall/knn-vc",
        "hifigan_wavlm",
        trust_repo=True,
        prematched=True,
        device=device
    )
    linearvc_model = linearvc.LinearVC(wavlm, hifigan, device)

    # -------------------------
    # load transform
    # -------------------------

    transforms = np.load(cfg["training"]["content_factorization_file"], allow_pickle=True).item()
    transform = torch.tensor(np.linalg.pinv(transforms[list(transforms.keys())[0]])).to(device)
    transform_tgt = torch.tensor(transforms[args.target_speaker]).to(device)

    # -------------------------
    # tokenize text
    # -------------------------

    tokens = [sp.encode_as_ids(args.text)]

    # -------------------------
    # flow matching sampling
    # -------------------------

    print("Running sampling...")
    with torch.no_grad():
        out_feats = model.sample(
            tokens=tokens,
            prompt_tokens=[[]],
            prompt_features=torch.tensor([[[]]], device=device),
            prompt_features_lens=torch.tensor([0], device=device),
            duration='real',
            features_lens=torch.tensor([540], device=device),
            num_step=int(args.sampling_steps)
        )[0]

    # -------------------------
    # vocode
    # -------------------------

    print("Running HiFiGAN...")
    with torch.no_grad():
        out_feats = invert_normalized_input(out_feats)
        audio = linearvc_model.hifigan(torch.matmul(out_feats, transform_tgt))

    audio = audio.squeeze().cpu()

    # -------------------------
    # save
    # -------------------------

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(args.out, audio.unsqueeze(0), 16000)

    print("Saved:", args.out)


if __name__ == "__main__":
    main()
