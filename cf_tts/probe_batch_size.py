"""Find the largest safe full-training batch of synthetic 30-second clips."""

import argparse
import gc
from pathlib import Path

import torch

from linearvc.cf_tts.models.tts import ZipVoice
from linearvc.cf_tts.utils.common import load_config, load_content_projection, normalize_input


def is_oom(error: BaseException) -> bool:
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()


def load_content_transform(config, device):
    factorization = config["training"]["content_factorization"]
    if factorization["type"] != "content":
        raise ValueError("The batch-size probe currently supports content factorization only.")
    return load_content_projection(factorization, device)


def try_batch(model, optimizer, wavlm, transform, config, batch_size, device):
    num_audio_samples = int(
        config["data"]["sampling_rate"]
        * config["data"]["train"]["streaming_reference_audio_seconds"]
    )
    token_count = config["data"]["train"]["streaming_probe_tokens_per_clip"]
    wavs = input_features = wav_lengths = loss = None
    try:
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        wavs = torch.zeros(batch_size, num_audio_samples, device=device)
        with torch.no_grad():
            input_features, _ = wavlm.extract_features(wavs, output_layer=6)
            input_features = torch.matmul(input_features, transform)
            input_features = input_features * config["training"]["feature_scale"]
            if config["training"]["normalize_input"]:
                input_features = normalize_input(input_features)
            input_features = input_features.detach()
            wav_lengths = torch.full(
                (batch_size,),
                input_features.shape[1],
                dtype=torch.int64,
                device=device,
            )

        loss = model(
            tokens=[[1] * token_count for _ in range(batch_size)],
            features=input_features,
            features_lens=wav_lengths,
            noise=config["training"]["noise_scale"] * torch.randn_like(input_features),
            t=torch.rand(batch_size, 1, 1, device=device),
            accents=[],
            condition_drop_ratio=config["training"]["condition_drop_ratio"],
        )
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        return True, torch.cuda.max_memory_allocated(device)
    except RuntimeError as error:
        if not is_oom(error):
            raise
        return False, None
    finally:
        del wavs, input_features, wav_lengths, loss
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the batch-size probe.")

    config = load_config(args.config)
    device = torch.device("cuda")
    probe_config = config["data"]["train"]
    max_batch_size = probe_config["streaming_probe_max_batch_size"]

    model = ZipVoice(**config["model"]["tts"]["zipvoice"]).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["optim"]["lr"],
        weight_decay=config["optim"]["weight_decay"],
    )
    wavlm = torch.hub.load(
        "bshall/knn-vc",
        "wavlm_large",
        trust_repo=True,
        progress=True,
        device=device,
    )
    # Training keeps HiFiGAN resident even though this path does not invoke it.
    hifigan, _ = torch.hub.load(
        "bshall/knn-vc",
        "hifigan_wavlm",
        trust_repo=True,
        prematched=True,
        progress=True,
        device=device,
    )
    transform = load_content_transform(config, device)

    low = 0
    high = None
    candidate = 1
    while candidate <= max_batch_size:
        ok, peak = try_batch(model, optimizer, wavlm, transform, config, candidate, device)
        if not ok:
            high = candidate
            print(f"PROBE batch={candidate} result=oom", flush=True)
            break
        low = candidate
        print(f"PROBE batch={candidate} result=ok peak_gib={peak / 1024**3:.2f}", flush=True)
        candidate *= 2

    if low == 0:
        raise RuntimeError("A batch size of one 30-second clip does not fit on this GPU.")
    if high is None:
        print(f"SAFE_BATCH_SIZE={low}", flush=True)
        return

    while low + 1 < high:
        candidate = (low + high) // 2
        ok, peak = try_batch(model, optimizer, wavlm, transform, config, candidate, device)
        if ok:
            low = candidate
            print(f"PROBE batch={candidate} result=ok peak_gib={peak / 1024**3:.2f}", flush=True)
        else:
            high = candidate
            print(f"PROBE batch={candidate} result=oom", flush=True)

    print(f"SAFE_BATCH_SIZE={low}", flush=True)


if __name__ == "__main__":
    main()
