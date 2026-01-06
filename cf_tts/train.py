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
from linearvc.cf_tts.utils.common import create_grad_scaler


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
    parser.add_argument("--checkpoint_path", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(cfg)

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
    if args.checkpoint_path is not None:
        current_step = load_checkpoint(model, args.checkpoint_path, device)
    else:
        current_step = 0
    current_epoch = current_step / len(train_loader)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"]
    )

    scaler = create_grad_scaler()

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
        if epoch < current_epoch:
            continue

        print(f"\nEpoch {epoch}")

        for batch_index, batch in enumerate(train_loader):
            if step < (current_step - current_epoch * len(train_loader)):
                continue
            if batch_index > 0:
                break
            wavs = batch["wav"].to(device)            # (B, T)
            wav_lengths = batch["wav_lengths"]
            texts = batch["text"]                     # list[str]
            speakers = batch["speaker"]               # list[str]

            text_ids = []
            for text in texts:
                text_ids.append(sp.encode_as_ids(text))

            with torch.no_grad():
                input_features, _ = linearvc_model.wavlm.extract_features(wavs, output_layer=6)
                input_features = torch.matmul(input_features, transform) * cfg['training']['feature_scale']

            wav_lengths = (torch.floor((wav_lengths - 400) / 320) + 1).to(device)

            # forward
            loss = model(
                tokens=text_ids,
                features=input_features,
                features_lens=wav_lengths,
                noise=cfg['training']['noise_scale'] * torch.randn_like(input_features).to(device), # Note: noise added to features. Not uniform random noise
                t=torch.rand(input_features.shape[0], 1, 1, device=device),
                condition_drop_ratio=cfg['training']['condition_drop_ratio']
            )

            optimizer.zero_grad()
            scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            if step % 1 == 0:
                print(f"epoch {(step / len(train_loader)):.3f} | total step {step} | loss {(loss).item():.4f}")

            if step % cfg["training"]["save_every"] == 0 and step > 0:
                save_checkpoint(
                    model,
                    optimizer,
                    step,
                    outdir / f"ckpt_{step}.pt",
                )

            step += 1
        
        # if step >= current_step:
        #     print(f"epoch {epoch} | loss {loss.item():.4f}")
        #     save_checkpoint(
        #         model,
        #         optimizer,
        #         step,
        #         outdir / f"ckpt_{step}.pt",
        #     )


if __name__ == "__main__":
    main()
