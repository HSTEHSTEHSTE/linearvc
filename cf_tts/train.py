import argparse, math
from pathlib import Path
import yaml

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import detect_anomaly

import sentencepiece as spm
import time, datetime
from linearvc import linearvc
from linearvc.cf_tts.dataset import TTSDataset, FrameBatchSampler, tts_collate
from linearvc.cf_tts.models.tts import ZipVoice
from linearvc.cf_tts.utils.common import create_grad_scaler, normalize_input


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

def load_checkpoint(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["step"]

def format_time(start_time, end_time):
    td = datetime.timedelta(seconds=(end_time - start_time))
    return str(td)

# -------------------------
# main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint_path", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(cfg, flush=True)

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

    if args.checkpoint_path is not None:
        current_step = load_checkpoint(model, optimizer, args.checkpoint_path, device)
        current_epoch = math.floor((current_step) / len(train_loader))
    else:
        current_step = -1
        current_epoch = 0

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

    step = current_epoch * len(train_loader)
    model.train()
    start_time = time.time()

    for epoch in range(cfg["training"]["epochs"]):
        losses = []
        if epoch < current_epoch:
            continue

        print(f"\nEpoch {epoch}", flush=True)

        for batch_index, batch in enumerate(train_loader):
            if step < current_step:
                step += 1
                continue
            if cfg['training']['epoch_batch_limit'] > 0 and batch_index >= cfg['training']['epoch_batch_limit']:
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
                input_features = torch.matmul(input_features, transform) # * cfg['training']['feature_scale']
                input_features = normalize_input(input_features)
                input_features = input_features.detach()

            wav_lengths = (torch.floor((wav_lengths - 400) / 320) + 1).to(device)

            # forward
            torch.manual_seed(step)
            loss = model(
                tokens=text_ids,
                features=input_features,
                features_lens=wav_lengths,
                noise=cfg['training']['noise_scale'] * torch.randn_like(input_features).to(device), # Note: noise added to features. Not uniform random noise
                t=torch.rand(input_features.shape[0], 1, 1, device=device),
                condition_drop_ratio=cfg['training']['condition_drop_ratio']
            )

            optimizer.zero_grad()
            if cfg['training']['use_grad_scaler']:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (list(model.parameters())[0].grad.isnan().any()):
                breakpoint()
            losses.append(loss.item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['clip_grad_norm'])
            optimizer.step()

            if step % cfg['training']['log_every'] == 0:
                current_time = time.time()
                print(f"epoch {(step / len(train_loader)):.3f} | time elapsed {format_time(start_time, current_time)} | total step {step} | loss {(loss).item():.4f}", flush=True)

            step += 1

            if step % cfg["training"]["save_every"] == 0 and step > 0:
                save_checkpoint(
                    model,
                    optimizer,
                    step,
                    outdir / f"ckpt_loss_{(loss).item():.2f}_step_{step}.pt",
                )

        
        current_time = time.time()
        print(f"epoch {epoch} | time elapsed {format_time(start_time, current_time)} | avg loss {(sum(losses) / len(losses)):.4f}")
        save_checkpoint(
            model,
            optimizer,
            step,
            outdir / f"ckpt_loss_{(sum(losses) / len(losses)):.2f}_epoch_{epoch}.pt",
        )


if __name__ == "__main__":
    main()
