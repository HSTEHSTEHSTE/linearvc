import argparse
from pathlib import Path
import yaml

import numpy as np
import torch
from torch.utils.data import DataLoader

import sentencepiece as spm
from linearvc import linearvc
from linearvc.cf_tts.dataset import TTSDataset, FrameBatchSampler, tts_collate
from linearvc.cf_tts.models.tts import ZipVoice


# -------------------------
# helpers
# -------------------------

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_checkpoint(model, optimizer, step, path):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        },
        path,
    )


# -------------------------
# main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # dataset
    # -------------------------

    train_set = TTSDataset(args.config)
    sampler = FrameBatchSampler(train_set, args.config)

    # since your dataset is already sorted by length,
    # do NOT shuffle here
    train_loader = DataLoader(
        train_set,
        batch_sampler=sampler,
        collate_fn=tts_collate,
        num_workers=cfg['training']['num_workers'],
        pin_memory=True,
    )

    # -------------------------
    # model
    # -------------------------

    model = ZipVoice(
        **cfg["model"]["tts"]["zipvoice"]
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"]
    )

    sp = spm.SentencePieceProcessor()
    sp.load(cfg['training']['spm_file'])

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

    transform = np.load(cfg['training']['content_factorization_file'], allow_pickle=True).item()
    transform = torch.tensor(np.linalg.pinv(transform[list(transform.keys())[0]])).to(device)

    # -------------------------
    # training loop
    # -------------------------

    outdir = Path(cfg["training"]["out_dir"])
    outdir.mkdir(parents=True, exist_ok=True)

    step = 0
    model.train()

    for epoch in range(cfg["training"]["epochs"]):
        print(f"\nEpoch {epoch}")

        for batch in train_loader:
            wavs = batch["wav"].to(device)            # (B, T)
            wav_lengths = batch["wav_lengths"]
            texts = batch["text"]                     # list[str]
            speakers = batch["speaker"]               # list[str]

            text_ids = []
            for text in texts:
                text_ids.append(sp.encode_as_ids(text))

            input_features, _ = linearvc_model.wavlm.extract_features(wavs, output_layer=6)
            input_features = torch.matmul(input_features, transform)

            wav_lengths = torch.floor(wav_lengths / 320).to(device)

            # forward
            outputs = model(
                tokens=text_ids,
                features=wav,
                features_lens=wav_lengths,

            )

            # assume model returns dict with loss
            loss = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % 50 == 0:
                print(f"step {step} | loss {loss.item():.4f}")

            if step % cfg["training"]["save_every"] == 0 and step > 0:
                save_checkpoint(
                    model,
                    optimizer,
                    step,
                    outdir / f"ckpt_{step}.pt",
                )

            step += 1


if __name__ == "__main__":
    main()
