#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, time, random, argparse, math
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# --------------------- data utils (meow) ---------------------
def utt_to_spk(utt_id: str) -> str:
    return utt_id.split("-")[0]

def seconds_to_ms(x: float) -> int:
    return int(round(1000.0 * x))

def make_pad_mask(lengths, max_len=None):
    B = lengths.size(0)
    T = int(max_len or lengths.max().item())
    idx = torch.arange(T, device=lengths.device).unsqueeze(0).expand(B, -1)
    return idx < lengths.unsqueeze(1)  # True valid

def build_vocab(alignment_dict, token_field="phoneme",
                min_freq=1, add_special=True):
    """
    alignment_dict[utt] is a LIST of dicts:
      {'start': float, 'end': float, 'phoneme': str}
    """
    c = Counter()
    for ex in alignment_dict.values():
        for ph in ex:
            c[ph[token_field]] += 1
    tokens = sorted([t for t, f in c.items() if f >= min_freq])

    itos, stoi = [], {}
    if add_special:
        for sp in ["<pad>", "<unk>"]:
            stoi[sp] = len(itos); itos.append(sp)
    for t in tokens:
        if t not in stoi:
            stoi[t] = len(itos); itos.append(t)
    return stoi, itos

def build_speaker_map(utt_ids):
    spks = sorted({utt_to_spk(u) for u in utt_ids})
    spk2id = {s: i for i, s in enumerate(spks)}
    id2spk = {i: s for s, i in spk2id.items()}
    return spk2id, id2spk


class LibriAlignmentDurationDataset(Dataset):
    def __init__(self, alignment, utt_ids, stoi, spk2id,
                 token_field="phoneme",
                 clamp_max_ms=2000):
        self.alignment = alignment
        self.utt_ids = list(utt_ids)
        self.stoi = stoi
        self.spk2id = spk2id
        self.token_field = token_field
        self.clamp_max_ms = clamp_max_ms

    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, idx):
        utt = self.utt_ids[idx]
        ex = self.alignment[utt]  # list of phoneme dicts

        spk = utt_to_spk(utt)
        spk_id = self.spk2id[spk]

        ids, durs = [], []
        for ph in ex:
            lab = ph[self.token_field]
            tid = self.stoi.get(lab, self.stoi.get("<unk>", 1))
            d_ms = seconds_to_ms(ph["end"] - ph["start"])
            d_ms = max(0, min(self.clamp_max_ms, d_ms))
            ids.append(tid)
            durs.append(d_ms)

        return (
            torch.tensor(ids, dtype=torch.long),   # tokens [T]
            torch.tensor(durs, dtype=torch.long),  # dur_ms [T]
            torch.tensor(spk_id, dtype=torch.long) # spk_id []
        )

def collate_batch(batch, pad_id=0):
    B = len(batch)
    lengths = torch.tensor([b[0].numel() for b in batch], dtype=torch.long)
    T = int(lengths.max().item())

    tokens = torch.full((B, T), pad_id, dtype=torch.long)
    durs   = torch.zeros((B, T), dtype=torch.long)
    spk_id = torch.stack([b[2] for b in batch], dim=0)  # [B]

    for i, (tok, dur, sid) in enumerate(batch):
        n = tok.numel()
        tokens[i, :n] = tok
        durs[i, :n] = dur

    mask = make_pad_mask(lengths, max_len=T)         # True valid
    src_key_padding_mask = ~mask                     # True pad for Transformer
    return tokens, durs, spk_id, lengths, mask, src_key_padding_mask


# --------------------- model: speaker-conditioned FastSpeech2-style duration predictor (meow) ---------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # [max_len, d_model]

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


class TinyEncoder(nn.Module):
    """
    Phoneme encoder. This is analogous to the text/phoneme encoder whose hidden
    states feed FastSpeech2's variance adaptor. Meow.
    """
    def __init__(self, vocab_size, hidden=192, n_heads=4, n_layers=2,
                 dropout=0.1, max_len=10000):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden)
        self.pos = SinusoidalPositionalEncoding(hidden, max_len=max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            batch_first=True,
            activation="relu"
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, tokens, src_key_padding_mask=None):
        x = self.emb(tokens)  # [B,T,H]
        x = self.pos(x)
        return self.tr(x, src_key_padding_mask=src_key_padding_mask)


class ConvPredictorBlock(nn.Module):
    """
    One FastSpeech2-style variance predictor block:
      Conv1d -> ReLU -> LayerNorm -> Dropout

    Input/output tensors are batch-first [B,T,C]. Purr.
    """
    def __init__(self, in_dim, out_dim, kernel_size=3, dropout=0.5):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=kernel_size,
            padding=padding
        )
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # [B,T,C] -> [B,C,T] -> conv -> [B,T,C]
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.relu(x)
        x = self.ln(x)
        x = self.dropout(x)
        return x


class FastSpeech2DurationPredictor(nn.Module):
    """
    FastSpeech2-style duration predictor:
      encoder hidden states
        -> Conv1D + ReLU + LayerNorm + Dropout
        -> Conv1D + ReLU + LayerNorm + Dropout
        -> Linear
        -> log(1 + duration_ms)

    Non-autoregressive: predicts all phoneme durations in parallel. Meow.
    """
    def __init__(self, hidden=192, predictor_hidden=256,
                 kernel_size=3, dropout=0.5):
        super().__init__()
        self.block1 = ConvPredictorBlock(
            in_dim=hidden,
            out_dim=predictor_hidden,
            kernel_size=kernel_size,
            dropout=dropout
        )
        self.block2 = ConvPredictorBlock(
            in_dim=predictor_hidden,
            out_dim=predictor_hidden,
            kernel_size=kernel_size,
            dropout=dropout
        )
        self.proj = nn.Linear(predictor_hidden, 1)

    def forward(self, encoder_out, mask_valid=None):
        """
        encoder_out: [B,T,H]
        mask_valid:  [B,T] True valid, False pad
        returns:     [B,T] predicted log(1 + duration_ms)
        """
        x = self.block1(encoder_out)
        x = self.block2(x)
        log_dur = self.proj(x).squeeze(-1)  # [B,T]

        if mask_valid is not None:
            log_dur = log_dur.masked_fill(~mask_valid, 0.0)
        return log_dur


class SpeakerConditionedFastSpeech2DurationModel(nn.Module):
    """
    Speaker-conditioned version:
      phoneme tokens -> Transformer encoder -> + speaker embedding
      -> FastSpeech2-style convolutional duration predictor.

    This keeps the speaker conditioning from your old model, but replaces the
    autoregressive Transformer decoder with a parallel FS2 predictor. Mrrp.
    """
    def __init__(self, vocab_size, n_speakers, hidden=192,
                 enc_layers=2, n_heads=4, dropout=0.1,
                 predictor_hidden=256, predictor_kernel_size=3,
                 predictor_dropout=0.5):
        super().__init__()
        self.enc = TinyEncoder(
            vocab_size=vocab_size,
            hidden=hidden,
            n_heads=n_heads,
            n_layers=enc_layers,
            dropout=dropout
        )
        self.spk = nn.Embedding(n_speakers, hidden)
        self.dur_pred = FastSpeech2DurationPredictor(
            hidden=hidden,
            predictor_hidden=predictor_hidden,
            kernel_size=predictor_kernel_size,
            dropout=predictor_dropout
        )

    def forward(self, tokens, spk_id, src_key_padding_mask=None, mask=None):
        memory = self.enc(tokens, src_key_padding_mask=src_key_padding_mask)  # [B,T,H]
        memory = memory + self.spk(spk_id).unsqueeze(1)  # speaker conditioning, meow
        return self.dur_pred(memory, mask_valid=mask)


def duration_loss(log_dur_pred, dur_ms, mask):
    log_gt = torch.log(dur_ms.float() + 1.0)
    mse = (log_dur_pred - log_gt) ** 2
    return mse.masked_select(mask).mean()


# --------------------- train (meow) ---------------------
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def evaluate(model, dl, device, desc="valid"):
    model.eval()
    tot, n = 0.0, 0
    pbar = tqdm(dl, desc=f"meow {desc}", leave=False, dynamic_ncols=True)
    for tokens, durs, spk_id, lengths, mask, src_kpm in pbar:
        tokens = tokens.to(device)
        durs = durs.to(device)
        spk_id = spk_id.to(device)
        mask = mask.to(device)
        src_kpm = src_kpm.to(device)

        log_dur = model(
            tokens,
            spk_id,
            src_key_padding_mask=src_kpm,
            mask=mask
        )
        loss = duration_loss(log_dur, durs, mask)
        tot += float(loss)
        n += 1
        pbar.set_postfix(loss=f"{tot/max(1,n):.4f}")
    return tot / max(1, n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--align_json", type=str, required=True,
                    help="Path to alignment.json (dict keyed by utt/file-stem; values are list of phoneme dicts).")
    ap.add_argument("--outdir", type=str, default="runs/durpred_fs2_spk")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--valid_split", type=float, default=0.01)
    ap.add_argument("--save_every_steps", type=int, default=500)
    ap.add_argument("--resume", type=str, default=None, help="checkpoint .pt to resume")

    ap.add_argument("--enc_layers", type=int, default=2)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1,
                    help="Dropout used in the Transformer phoneme encoder.")

    ap.add_argument("--predictor_hidden", type=int, default=256,
                    help="Hidden/channel size of the FastSpeech2 duration predictor.")
    ap.add_argument("--predictor_kernel_size", type=int, default=3,
                    help="Conv1d kernel size in the FastSpeech2 duration predictor.")
    ap.add_argument("--predictor_dropout", type=float, default=0.5,
                    help="Dropout used in the FastSpeech2 duration predictor.")

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    print("meow loading alignments:", args.align_json, flush=True)
    with open(args.align_json, "r") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "alignments_wb" in raw:
        alignment = raw["alignments_wb"]
    else:
        alignment = raw

    utt_ids = list(alignment.keys())
    random.shuffle(utt_ids)

    n_valid = max(1, int(len(utt_ids) * args.valid_split))
    valid_ids = utt_ids[:n_valid]
    train_ids = utt_ids[n_valid:]

    stoi, itos = build_vocab(alignment, token_field="phoneme")
    spk2id, id2spk = build_speaker_map(utt_ids)

    meta = {
        "vocab_size": len(itos),
        "n_speakers": len(spk2id),
        "stoi": stoi,
        "itos": itos,
        "spk2id": spk2id,
        "id2spk": id2spk,
        "token_field": "phoneme",
        "dur_unit": "ms",
        "target": "log(1 + duration_ms)",
        "model": "SpeakerConditionedFastSpeech2DurationModel",
        "architecture": (
            "Transformer phoneme encoder + speaker embedding + "
            "FastSpeech2-style non-autoregressive Conv1D duration predictor"
        ),
        "predictor": {
            "blocks": 2,
            "block": "Conv1d -> ReLU -> LayerNorm -> Dropout",
            "predictor_hidden": args.predictor_hidden,
            "predictor_kernel_size": args.predictor_kernel_size,
            "predictor_dropout": args.predictor_dropout
        }
    }
    with open(os.path.join(args.outdir, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"meow wrote meta.json (vocab={len(itos)} speakers={len(spk2id)})", flush=True)

    train_ds = LibriAlignmentDurationDataset(
        alignment, train_ids, stoi, spk2id, token_field="phoneme"
    )
    valid_ds = LibriAlignmentDurationDataset(
        alignment, valid_ids, stoi, spk2id, token_field="phoneme"
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_batch(b, pad_id=stoi["<pad>"])
    )
    valid_dl = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_batch(b, pad_id=stoi["<pad>"])
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("meow device:", device, flush=True)

    model = SpeakerConditionedFastSpeech2DurationModel(
        vocab_size=len(itos),
        n_speakers=len(spk2id),
        hidden=args.hidden,
        enc_layers=args.enc_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        predictor_hidden=args.predictor_hidden,
        predictor_kernel_size=args.predictor_kernel_size,
        predictor_dropout=args.predictor_dropout
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    global_step = 0
    best_valid = float("inf")

    if args.resume:
        print("meow resuming from:", args.resume, flush=True)
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        for pg in opt.param_groups:
            pg["lr"] = args.lr
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)
        best_valid = ckpt.get("best_valid", best_valid)
        print(f"meow resumed epoch={start_epoch} step={global_step}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()

        pbar = tqdm(
            train_dl,
            desc=f"meow train epoch {epoch+1}/{args.epochs}",
            dynamic_ncols=True
        )

        running = 0.0
        running_n = 0

        for tokens, durs, spk_id, lengths, mask, src_kpm in pbar:
            tokens = tokens.to(device)
            durs = durs.to(device)
            spk_id = spk_id.to(device)
            mask = mask.to(device)
            src_kpm = src_kpm.to(device)

            log_dur = model(
                tokens,
                spk_id,
                src_key_padding_mask=src_kpm,
                mask=mask
            )
            loss = duration_loss(log_dur, durs, mask)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            global_step += 1
            running += float(loss)
            running_n += 1

            pbar.set_postfix(step=global_step, loss=f"{running/max(1,running_n):.4f}")

            if global_step % 1000 == 0:
                avg = running / max(1, running_n)
                print(
                    f"meow epoch {epoch+1}/{args.epochs} step {global_step} train_loss {avg:.4f}",
                    flush=True
                )
                running, running_n = 0.0, 0

            if global_step % args.save_every_steps == 0:
                val = evaluate(model, valid_dl, device, desc=f"valid@{global_step}")

                if val < best_valid:
                    best_valid = val
                    best_path = os.path.join(args.outdir, "best.pt")
                    torch.save({
                        "model": model.state_dict(),
                        "opt": opt.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "valid_loss": val,
                        "best_valid": best_valid,
                        "args": vars(args),
                    }, best_path)
                    print("meow new best saved", best_path, flush=True)

        val = evaluate(model, valid_dl, device, desc=f"valid_epoch{epoch+1}")
        dt = time.time() - t0
        print(f"meow epoch done {epoch+1}: valid_loss {val:.4f} time {dt:.1f}s", flush=True)

        ep_path = os.path.join(args.outdir, f"epoch{epoch+1}.pt")
        torch.save({
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "epoch": epoch + 1,
            "global_step": global_step,
            "valid_loss": val,
            "best_valid": best_valid,
            "args": vars(args),
        }, ep_path)
        print("meow saved", ep_path, flush=True)

    print("meow training complete. best_valid:", best_valid, flush=True)


if __name__ == "__main__":
    main()