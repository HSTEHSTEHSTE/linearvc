"""Find the largest full-training batch with a configurable GPU memory margin."""

import argparse
import gc
import math
from pathlib import Path

import torch

from linearvc.cf_tts.models.tts import ZipVoice
from linearvc.cf_tts.utils.common import load_config, load_content_projection, normalize_input
from linearvc.cf_tts.utils.artifacts import configure_artifact_environment


def is_oom(error: BaseException) -> bool:
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()


def load_content_transform(config, device):
    factorization = config["training"]["content_factorization"]
    if factorization["type"] != "content":
        raise ValueError("The batch-size probe currently supports content factorization only.")
    return load_content_projection(factorization, device)


def try_batch(model, optimizer, wavlm, transform, config, batch_size, token_count, device):
    num_audio_samples = int(
        config["data"]["sampling_rate"]
        * config["data"]["train"]["streaming_reference_audio_seconds"]
    )
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
        return (
            True,
            torch.cuda.max_memory_allocated(device),
            torch.cuda.max_memory_reserved(device),
        )
    except RuntimeError as error:
        if not is_oom(error):
            raise
        return False, None, None
    finally:
        del wavs, input_features, wav_lengths, loss
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()


def probe_token_lengths(config):
    """Return text lengths that cover every permitted padded-text regime."""
    train_config = config["data"]["train"]
    budget = int(train_config["streaming_token_square_budget"])
    maximum = min(
        int(train_config["streaming_max_tokens_per_utterance"]),
        math.isqrt(budget),
    )
    if maximum < 1:
        raise ValueError("No text length can fit within streaming_token_square_budget.")
    baseline = min(int(train_config["streaming_probe_tokens_per_clip"]), maximum)
    lengths = {baseline, maximum}
    candidate = baseline
    while candidate < maximum:
        candidate = min(maximum, candidate * 2)
        lengths.add(candidate)
    return sorted(lengths)


def microbatch_size(reference_batch_size, token_count, token_square_budget):
    """Match the trainer's padded-token microbatch capacity."""
    return max(1, min(reference_batch_size, token_square_budget // (token_count * token_count)))


def try_reference_batch(
    model,
    optimizer,
    wavlm,
    transform,
    config,
    reference_batch_size,
    device,
    memory_safety_margin_fraction,
):
    """Test all allowed text-length profiles for one audio reference batch."""
    train_config = config["data"]["train"]
    budget = int(train_config["streaming_token_square_budget"])
    peak_allocated = 0
    peak_reserved = 0
    usable_memory = int(
        torch.cuda.get_device_properties(device).total_memory
        * (1.0 - memory_safety_margin_fraction)
    )
    for token_count in probe_token_lengths(config):
        batch_size = microbatch_size(reference_batch_size, token_count, budget)
        ok, profile_allocated, profile_reserved = try_batch(
            model, optimizer, wavlm, transform, config, batch_size, token_count, device
        )
        if not ok:
            print(
                f"PROBE reference_batch={reference_batch_size} text_tokens={token_count} "
                f"microbatch={batch_size} result=oom",
                flush=True,
            )
            return False, None, None
        peak_allocated = max(peak_allocated, profile_allocated)
        peak_reserved = max(peak_reserved, profile_reserved)
        print(
            f"PROBE reference_batch={reference_batch_size} text_tokens={token_count} "
            f"microbatch={batch_size} result=ok "
            f"allocated_gib={profile_allocated / 1024**3:.2f} "
            f"reserved_gib={profile_reserved / 1024**3:.2f}",
            flush=True,
        )
    if peak_reserved > usable_memory:
        print(
            f"PROBE reference_batch={reference_batch_size} result=insufficient_margin "
            f"peak_reserved_gib={peak_reserved / 1024**3:.2f} "
            f"usable_gib={usable_memory / 1024**3:.2f} "
            f"margin_fraction={memory_safety_margin_fraction:.3f}",
            flush=True,
        )
        return False, peak_allocated, peak_reserved
    return True, peak_allocated, peak_reserved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--memory-safety-margin-fraction",
        type=float,
        default=0.05,
        help="Keep this fraction of total GPU memory free after the worst probe profile (default: 0.05).",
    )
    args = parser.parse_args()

    if not 0.0 <= args.memory_safety_margin_fraction < 1.0:
        raise ValueError("memory-safety-margin-fraction must be in [0, 1).")

    configure_artifact_environment()

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
        ok, peak_allocated, peak_reserved = try_reference_batch(
            model, optimizer, wavlm, transform, config, candidate, device,
            args.memory_safety_margin_fraction,
        )
        if not ok:
            high = candidate
            break
        low = candidate
        print(
            f"PROBE reference_batch={candidate} result=ok "
            f"peak_allocated_gib={peak_allocated / 1024**3:.2f} "
            f"peak_reserved_gib={peak_reserved / 1024**3:.2f}",
            flush=True,
        )
        candidate *= 2

    if low == 0:
        raise RuntimeError("A batch size of one 30-second clip does not fit on this GPU.")
    if high is None:
        print(f"SAFE_BATCH_SIZE={low}", flush=True)
        return

    while low + 1 < high:
        candidate = (low + high) // 2
        ok, peak_allocated, peak_reserved = try_reference_batch(
            model, optimizer, wavlm, transform, config, candidate, device,
            args.memory_safety_margin_fraction,
        )
        if ok:
            low = candidate
            print(
                f"PROBE reference_batch={candidate} result=ok "
                f"peak_allocated_gib={peak_allocated / 1024**3:.2f} "
                f"peak_reserved_gib={peak_reserved / 1024**3:.2f}",
                flush=True,
            )
        else:
            high = candidate

    print(f"SAFE_BATCH_SIZE={low}", flush=True)


if __name__ == "__main__":
    main()
