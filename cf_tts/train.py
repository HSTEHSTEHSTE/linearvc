import argparse, math, shutil
from pathlib import Path
import wandb

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import detect_anomaly

import time
from linearvc import linearvc
from linearvc.cf_tts.dataset import BackgroundPrefetchDataset, EmiliaLengthSortedBatchDataset, EmiliaStreamingDataset, FrameBatchSampler, TTSDataset, TTS_Collate
from linearvc.cf_tts.models.tts import ZipVoice
from linearvc.cf_tts.utils.common import create_grad_scaler, invert_normalized_input, normalize_input, load_config, format_time, get_speaker_feats, load_content_projection, match_knn
from linearvc.cf_tts.utils.checkpoints import load_checkpoint, manage_checkpoints, save_checkpoint
from linearvc.cf_tts.utils.optim import get_scheduler

# -------------------------
# helpers
# -------------------------

def split_text_microbatches(tokens, token_square_budget, max_tokens_per_utterance):
    """Pack ordered examples into text-memory-safe GPU microbatches."""
    if token_square_budget < 1 or max_tokens_per_utterance < 1:
        raise ValueError("Text microbatch limits must be positive.")

    microbatches = []
    skipped = []
    current = []
    current_cost = 0
    for index, token_ids in enumerate(tokens):
        token_count = len(token_ids)
        token_cost = token_count * token_count
        if token_count > max_tokens_per_utterance or token_cost > token_square_budget:
            skipped.append((index, token_count))
            continue
        if current and current_cost + token_cost > token_square_budget:
            microbatches.append(current)
            current = []
            current_cost = 0
        current.append(index)
        current_cost += token_cost
    if current:
        microbatches.append(current)
    return microbatches, skipped


def train_target_frame_count(wav_lengths, cfg):
    """Number of WavLM frames scored outside the in-utterance style prompt."""
    if cfg['training']['content_factorization']['type'] == 'fbank':
        raise ValueError('Text microbatch weighting currently requires WavLM features.')
    feature_lengths = torch.floor((wav_lengths.to(torch.float64) - 400) / 320) + 1
    prompt_frames = int(cfg['model']['tts']['zipvoice']['style_prompt_frames'])
    return torch.clamp(feature_lengths.to(torch.int64) - prompt_frames, min=0).sum().item()


def plan_gradient_microbatches(physical_batches, cfg):
    """Plan memory-bounded GPU passes without concatenating physical batches.

    Each planned pass retains a reference to its original physical batch. This
    is essential: combining several safe physical batches before text packing
    could silently recreate an unsafe GPU batch when their transcripts are
    short.
    """
    token_square_budget = int(cfg['data']['train']['streaming_token_square_budget'])
    max_tokens_per_utterance = int(cfg['data']['train']['streaming_max_tokens_per_utterance'])
    plan = []
    skipped = []
    for physical_index, batch in enumerate(physical_batches):
        microbatches, skipped_texts = split_text_microbatches(
            batch['text'],
            token_square_budget=token_square_budget,
            max_tokens_per_utterance=max_tokens_per_utterance,
        )
        skipped.extend(
            (physical_index, index, token_count)
            for index, token_count in skipped_texts
        )
        for indices in microbatches:
            target_frame_count = train_target_frame_count(batch['wav_lengths'][indices], cfg)
            if target_frame_count:
                plan.append((batch, indices, target_frame_count))
    return plan, skipped


def accumulation_target_reached(
    physical_examples: int,
    physical_audio_seconds: float,
    cfg,
) -> bool:
    """Return whether a streaming optimizer update has enough audio.

    Audio-seconds take precedence over the legacy example-count target. This
    keeps the effective batch stable even when Emilia's 3--30 second segments
    cause the physical batch size to vary.
    """
    train_cfg = cfg['data']['train']
    target_seconds = float(train_cfg.get('streaming_gradient_accumulation_target_seconds', 0.0))
    if target_seconds < 0.0:
        raise ValueError('streaming_gradient_accumulation_target_seconds must be non-negative.')
    if target_seconds > 0.0:
        return physical_audio_seconds >= target_seconds

    target_examples = int(train_cfg.get('streaming_gradient_accumulation_target_examples', 0))
    if target_examples < 0:
        raise ValueError('streaming_gradient_accumulation_target_examples must be non-negative.')
    # Preserve the historical behavior: no configured target means one
    # physical batch per optimizer update.
    return target_examples == 0 or physical_examples >= target_examples


def backward_train_microbatch(
    model,
    linearvc_model,
    transform,
    cfg,
    micro_wavs,
    micro_wav_lengths,
    micro_text_ids,
    micro_accents,
    device,
    loss_weight,
    scaler,
):
    """Run one gradient-accumulated train microbatch and return its raw loss."""
    with torch.no_grad():
        if cfg['training']['content_factorization']['type'] == 'fbank':
            input_features = transform(micro_wavs)
        else:
            input_features, _ = linearvc_model.wavlm.extract_features(micro_wavs, output_layer=6)
            if cfg['training']['content_factorization']['type'] == 'content' and transform is not None:
                input_features = torch.matmul(input_features, transform)
            elif cfg['training']['content_factorization']['type'] == 'speaker':
                input_features = match_knn(input_features, transform)
        input_features = input_features * cfg['training']['feature_scale']
        if cfg['training']['normalize_input']:
            input_features = normalize_input(input_features)
        input_features = input_features.detach()

    if cfg['training']['content_factorization']['type'] == 'fbank':
        features_lens = torch.full([input_features.shape[0]], input_features.shape[1], device=device)
    else:
        features_lens = (torch.floor((micro_wav_lengths - 400) / 320) + 1).to(device)
    loss = model(
        tokens=micro_text_ids,
        features=input_features,
        features_lens=features_lens,
        accents=micro_accents,
        noise=cfg['training']['noise_scale'] * torch.randn_like(input_features).to(device),
        t=torch.rand(input_features.shape[0], 1, 1, device=device),
        condition_drop_ratio=cfg['training']['condition_drop_ratio'],
    )
    weighted_loss = loss * loss_weight
    if cfg['training']['use_grad_scaler']:
        scaler.scale(weighted_loss).backward()
    else:
        weighted_loss.backward()
    return loss.item()

def prepare_validation_features(wavs, cfg, linearvc_model, transform):
    """Extract the acoustic representation used by validation and sampling."""
    factorization_type = cfg['training']['content_factorization']['type']
    if factorization_type == 'fbank':
        input_features = transform(wavs)
        raw_wavlm_features = None
    else:
        raw_wavlm_features, _ = linearvc_model.wavlm.extract_features(wavs, output_layer=6)
        input_features = raw_wavlm_features
        if factorization_type == 'content' and transform is not None:
            input_features = torch.matmul(input_features, transform)
        elif factorization_type == 'speaker':
            input_features = match_knn(input_features, transform)

    input_features = input_features * cfg['training']['feature_scale']
    if cfg['training']['normalize_input']:
        input_features = normalize_input(input_features)
    return input_features.detach(), None if raw_wavlm_features is None else raw_wavlm_features.detach()


def build_fixed_validation_audio_batch(dev_set, tts_collate, cfg):
    """Load one reproducible target/prompt pair for qualitative validation.

    This is intentionally separate from the shuffled loss-validation loader.
    The configured target path pins the comparison utterance; without one, the
    lexicographically first usable Common Voice path is selected deterministically.
    """
    requested_path = cfg['data']['dev'].get('validation_audio_target_path')
    prompt_samples = int(3.0 * cfg['data']['sampling_rate'])
    max_audio_len = int(cfg['data']['max_audio_len'])

    indexed_data = list(enumerate(dev_set.data))
    if requested_path:
        requested_path = Path(requested_path).resolve()
        indexed_data = [
            (index, datum)
            for index, datum in indexed_data
            if datum.wav_path.resolve() == requested_path
        ]
        if not indexed_data:
            raise ValueError(f'Configured validation_audio_target_path was not found: {requested_path}')
    else:
        indexed_data.sort(key=lambda item: str(item[1].wav_path))

    for index, datum in indexed_data:
        if datum.prompt_wav_path is None or min(datum.num_frames, max_audio_len) <= prompt_samples:
            continue
        try:
            batch = tts_collate([dev_set[index]])
        except Exception as exc:
            print(f'Skipping fixed validation audio candidate {datum.wav_path}: {exc}', flush=True)
            continue
        if batch['prompt_available'] is not None and batch['prompt_available'][0] and batch['text'][0]:
            print(
                f'Fixed validation audio target: {datum.wav_path} | prompt: {datum.prompt_wav_path}',
                flush=True,
            )
            return batch

    raise RuntimeError('Could not find a promptable, transcribable fixed validation audio example.')


@torch.inference_mode()
def log_validation_audio(cfg, model, linearvc_model, transform, fixed_batch, device, step, epoch):
    """Synthesize the fixed target text using only its separate style prompt.

    No target waveform, target WavLM feature, or target duration is used in
    sampling or vocoding. For content factorization, the content-to-WavLM
    transform is estimated from the prompt itself.
    """
    try:
        prompt_wavs = fixed_batch['prompt_wav'].to(device)
        prompt_features, raw_prompt_wavlm_features = prepare_validation_features(
            prompt_wavs, cfg, linearvc_model, transform
        )
        prompt_len = int(prompt_features.shape[1])
        if not fixed_batch['text'][0]:
            raise ValueError('Fixed validation target has no text tokens.')

        duration_seconds = float(cfg['data']['dev'].get('validation_audio_duration_seconds', 8.0))
        generated_frames = max(1, round(duration_seconds * cfg['data']['sampling_rate'] / 320))
        prompt_lens = torch.tensor([prompt_len], dtype=torch.int64, device=device)
        generated_lens = torch.tensor([generated_frames], dtype=torch.int64, device=device)

        # Keep sampling noise fixed so a checkpoint's qualitative sample is
        # reproducible and does not perturb the training RNG stream.
        sample_seed = int(cfg['data']['dev'].get('validation_audio_seed', 42))
        cuda_devices = [device.index] if device.type == 'cuda' and device.index is not None else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(sample_seed)
            if device.type == 'cuda':
                torch.cuda.manual_seed(sample_seed)
            generated_features = model.sample(
                tokens=[fixed_batch['text'][0]],
                prompt_tokens=[[]],
                prompt_features=prompt_features,
                prompt_features_lens=prompt_lens,
                duration='real',
                features_lens=generated_lens,
                num_step=int(cfg['training']['sampling_steps']),
            )[0]

        if cfg['training']['normalize_input']:
            generated_features = invert_normalized_input(generated_features)
        generated_features = generated_features / cfg['training']['feature_scale']

        factorization_type = cfg['training']['content_factorization']['type']
        if factorization_type == 'content':
            if raw_prompt_wavlm_features is None:
                raise ValueError('Content-factorized validation audio requires prompt WavLM features.')
            prompt_content = prompt_features
            if cfg['training']['normalize_input']:
                prompt_content = invert_normalized_input(prompt_content)
            prompt_content = prompt_content / cfg['training']['feature_scale']
            prompt_transform = torch.matmul(
                torch.linalg.pinv(prompt_content), raw_prompt_wavlm_features[:, :prompt_len]
            )
            generated_features = torch.matmul(generated_features, prompt_transform)
        elif factorization_type == 'speaker':
            generated_features = match_knn(generated_features, transform)
        elif factorization_type == 'fbank':
            raise ValueError('HiFi-GAN validation audio is unsupported for fbank features.')

        audio = linearvc_model.hifigan(generated_features).reshape(-1).detach().cpu().numpy()
        target_name = Path(fixed_batch['wav_path'][0]).name
        prompt_name = Path(fixed_batch['prompt_wav_path'][0]).name
        wandb.log(
            {
                'validation_audio': wandb.Audio(
                    audio,
                    sample_rate=cfg['data']['sampling_rate'],
                    caption=(
                        f'epoch={epoch}, step={step}, target={target_name}, prompt={prompt_name}, '
                        f'prompt=3.0s, generated={duration_seconds:.1f}s, seed={sample_seed}'
                    ),
                ),
            },
            step=step,
        )
        return True
    except Exception as exc:
        print(f'Validation audio synthesis failed: {exc}', flush=True)
        return False


def validation_pass(dev_loader, fixed_audio_batch, linearvc_model, transform, cfg, model, device, step, epoch):
    model.eval()

    total_loss = 0.
    total = 0
    audio_logged = False
    # Keep these names defined so the final (often large) validation batch can
    # be released before qualitative sampling asks CUDA for more memory.
    batch = wavs = prompt_wavs = input_features = prompt_features = None
    wav_lengths = prompt_features_lens = valid_indices = None
    text_ids = accents = loss = None
    for batch_index, batch in enumerate(dev_loader):
        if cfg['data']['dev']['epoch_batch_limit'] > 0 and batch_index >= cfg['data']['dev']['epoch_batch_limit']:
            break
        prompt_available = batch['prompt_available']
        if prompt_available is None or not prompt_available.any():
            print(f'Skipping validation batch {batch_index}: no distinct same-speaker style prompt.', flush=True)
            continue

        prompt_indices = prompt_available.nonzero(as_tuple=False).squeeze(1)
        wav_lengths = batch['wav_lengths'][prompt_indices]
        # Advanced indexing keeps the original batch's padded width, which
        # may belong to an unpromptable row. Trim it so WavLM's time axis
        # agrees with the selected targets' true maximum length.
        wavs = batch['wav'][prompt_indices, : int(wav_lengths.max())].to(device)
        wav_lengths = wav_lengths.to(device)
        text_ids = [batch['text'][index] for index in prompt_indices.tolist()]
        accents = [batch['accent'][index] for index in prompt_indices.tolist()] if batch['accent'] else []
        prompt_wavs = batch['prompt_wav'][prompt_indices].to(device)

        with torch.no_grad():
            input_features, _ = prepare_validation_features(
                wavs, cfg, linearvc_model, transform
            )
            prompt_features, _ = prepare_validation_features(
                prompt_wavs, cfg, linearvc_model, transform
            )

            if cfg['training']['content_factorization']['type'] == 'fbank':
                wav_lengths = torch.full([input_features.shape[0]], input_features.shape[1]).to(device)
            else:
                wav_lengths = (torch.floor((wav_lengths - 400) / 320) + 1).to(device)
            prompt_features_lens = torch.full(
                [prompt_features.shape[0]], prompt_features.shape[1], dtype=torch.int64, device=device
            )
            valid_targets = wav_lengths > prompt_features_lens
            if not valid_targets.any():
                print(f'Skipping validation batch {batch_index}: no promptable utterance exceeds its prompt length.', flush=True)
                continue

            valid_indices = valid_targets.nonzero(as_tuple=False).squeeze(1)
            input_features = input_features[valid_indices]
            prompt_features = prompt_features[valid_indices]
            prompt_features_lens = prompt_features_lens[valid_indices]
            wav_lengths = wav_lengths[valid_indices]
            text_ids = [text_ids[index] for index in valid_indices.tolist()]
            accents = [accents[index] for index in valid_indices.tolist()] if accents else []

            loss = model(
                tokens=text_ids,
                features=input_features,
                features_lens=wav_lengths,
                noise=cfg['training']['noise_scale'] * torch.randn_like(input_features).to(device), # Note: noise added to features. Not uniform random noise
                t=torch.rand(input_features.shape[0], 1, 1, device=device),
                accents=accents,
                condition_drop_ratio=0.,
                prompt_features=prompt_features,
                prompt_features_lens=prompt_features_lens,
            )

        total_loss += loss.item()
        total += 1

    # Validation loss is already no-grad, but its final batch remains in this
    # function's locals. The 500M model needs that memory returned before the
    # separate W&B sampling pass starts.
    del batch, wavs, prompt_wavs, input_features, prompt_features
    del wav_lengths, prompt_features_lens, valid_indices, text_ids, accents, loss
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    if not audio_logged:
        audio_logged = log_validation_audio(
            cfg=cfg,
            model=model,
            linearvc_model=linearvc_model,
            transform=transform,
            fixed_batch=fixed_audio_batch,
            device=device,
            step=step,
            epoch=epoch,
        )

    if total == 0:
        raise RuntimeError('Validation contained no frames beyond the style prompt.')
    print("Validation")
    print(f"epoch {epoch} | total step {step} | val loss {(total_loss / total):.4f}", flush=True)
    model.train()
    return total_loss / total

# -------------------------
# main
# -------------------------

def main():
    def parse_bool(value):
        if isinstance(value, bool):
            return value
        if value.lower() in ('1', 'true', 'yes', 'y'):
            return True
        if value.lower() in ('0', 'false', 'no', 'n'):
            return False
        raise argparse.ArgumentTypeError(f'Expected a boolean value, got {value!r}.')

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint_path", default='')
    parser.add_argument("--ignore_current_step", type=parse_bool, default=False)
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(cfg, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.login()
    wandb.init(
        project=cfg['training']['project_name'],
        name=cfg['training']['run_name'],
        id=cfg['training']['run_name'],
    )

    # -------------------------
    # dataset
    # -------------------------

    train_is_streaming = cfg['data']['train']['dataset'] == 'emilia'
    dev_is_streaming = cfg['data']['dev']['dataset'] == 'emilia'

    load_path = None
    if train_is_streaming:
        if cfg['data']['train']['load_path'] is not None:
            raise ValueError("Emilia streaming does not support data.train.load_path; set it to null.")
        train_set = EmiliaLengthSortedBatchDataset(
            EmiliaStreamingDataset(config_file_path=args.config, split='train'),
            config_file_path=args.config,
            split='train',
        )
        train_set = BackgroundPrefetchDataset(
            train_set,
            prefetch_items=cfg['data']['train']['streaming_background_prefetch_batches'],
        )
        train_sampler = None
    elif cfg['data']['train']['load_path'] is not None:
        import pickle
        load_path = Path(cfg['data']['train']['load_path'])
        if load_path.is_file():
            print("Reading pre-processed data file", flush=True)
            with open(load_path, 'rb') as file:
                data_list = pickle.load(file)
            train_set = TTSDataset(config_file_path=args.config, split='train', data=data_list)
        else:
            print("Pre-processing data and saving to load_path", flush=True)
            train_set = TTSDataset(config_file_path=args.config, split='train')
            with open(load_path, 'wb') as file:
                pickle.dump(train_set.data, file)
    else:
        print("Pre-processing data", flush=True)
        train_set = TTSDataset(config_file_path=args.config, split='train')
    if not train_is_streaming:
        train_sampler = FrameBatchSampler(train_set, args.config, split='train')

    if dev_is_streaming:
        if cfg['data']['dev']['load_path'] is not None:
            raise ValueError("Emilia streaming does not support data.dev.load_path; set it to null.")
        dev_set = EmiliaLengthSortedBatchDataset(
            EmiliaStreamingDataset(config_file_path=args.config, split='dev'),
            config_file_path=args.config,
            split='dev',
        )
        dev_set = BackgroundPrefetchDataset(
            dev_set,
            prefetch_items=cfg['data']['dev']['streaming_background_prefetch_batches'],
        )
        dev_sampler = None
    elif cfg['data']['dev']['load_path'] is not None:
        import pickle
        with open(cfg['data']['dev']['load_path'], 'rb') as file:
            data_list = pickle.load(file)['data']
        dev_set = TTSDataset(config_file_path=args.config, split='dev', data=data_list)
    else:
        dev_set = TTSDataset(config_file_path=args.config, split='dev')
    if not dev_is_streaming:
        dev_sampler = FrameBatchSampler(dev_set, args.config, split='dev')
    

    tts_collate = TTS_Collate(cfg)
    if dev_is_streaming:
        raise ValueError('Qualitative validation audio requires a finite validation dataset.')
    fixed_audio_batch = build_fixed_validation_audio_batch(dev_set, tts_collate, cfg)
    if train_is_streaming:
        train_loader = DataLoader(
            train_set,
            batch_size=None,
            collate_fn=tts_collate,
            # Emilia prefetches in BackgroundPrefetchDataset; worker processes
            # would create POSIX semaphores on the Slurm node.
            num_workers=0,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_set,
            batch_sampler=train_sampler,
            collate_fn=tts_collate,
            num_workers=cfg['training']['num_workers'],
            pin_memory=True,
        )
    if dev_is_streaming:
        dev_loader = DataLoader(
            dev_set,
            batch_size=None,
            collate_fn=tts_collate,
            num_workers=0,
            pin_memory=True,
        )
    else:
        dev_loader = DataLoader(
            dev_set,
            batch_sampler=dev_sampler,
            collate_fn=tts_collate,
            # Validation is infrequent and must not depend on POSIX
            # semaphores, which can be exhausted on shared Slurm nodes.
            num_workers=0,
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

    if train_is_streaming:
        # A positive limit defines a synthetic epoch. Zero exhausts the
        # streaming dataset, i.e. one complete pass over all selected shards.
        epoch_batch_length = cfg['data']['train']['epoch_batch_limit'] or None
    elif cfg['data']['train']['epoch_batch_limit'] > 0:
        epoch_batch_length = min(len(train_loader), cfg['data']['train']['epoch_batch_limit'])
    else:
        epoch_batch_length = len(train_loader)

    checkpoint_state = None
    stream_resume_cursor = None
    if args.checkpoint_path and args.checkpoint_path != 'none':
        checkpoint_state = load_checkpoint(
            model,
            args.checkpoint_path,
            device,
            optimizer=optimizer,
            return_checkpoint=True,
        )
        current_step = checkpoint_state["step"]
        if epoch_batch_length is None:
            stream_resume_cursor = checkpoint_state.get("data_cursor")
            current_epoch = int(stream_resume_cursor.get("epoch", 0)) if stream_resume_cursor else 0
            if stream_resume_cursor:
                train_set.set_resume_cursor(stream_resume_cursor)
                print(
                    "Resuming Emilia stream at "
                    f"epoch {current_epoch}, shard {stream_resume_cursor['shard_index']}, "
                    f"batch {stream_resume_cursor['batch_index']}.",
                    flush=True,
                )
            else:
                print(
                    "Checkpoint has no Emilia stream cursor; starting a fresh dataset sweep.",
                    flush=True,
                )
        else:
            current_epoch = math.floor((current_step) / epoch_batch_length)
    else:
        current_step = -1
        current_epoch = 0
    if args.ignore_current_step:
        current_step = -1
        current_epoch = 0
        stream_resume_cursor = None
        if train_is_streaming:
            train_set.set_resume_cursor(None)

    scheduler = get_scheduler(cfg['optim']['scheduler_type'], optimizer, -1, cfg['optim']['scheduler_args'])
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Current learning rate: {current_lr}")

    scaler = create_grad_scaler()

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

    if cfg['training']['content_factorization']['type'] == 'content':
        if cfg['training']['content_factorization']['content_factorization_file'] is not None:
            transform = load_content_projection(
                cfg['training']['content_factorization'], device
            )
        else:
            transform = None
    elif cfg['training']['content_factorization']['type'] == 'speaker':
        from cuvs.neighbors import brute_force
        feats = get_speaker_feats(
            tgt_speaker_root=cfg['training']['content_factorization']['factorization_speaker'],
            linearvc_model=linearvc_model,
            device=device
        )
        index = brute_force.build(feats)
        transform = {
            'feats': feats,
            'index': index,
            'brute_force': brute_force
        }
    elif cfg['training']['content_factorization']['type'] == 'none':
        transform = None
    elif cfg['training']['content_factorization']['type'] == 'fbank':
        from speechbrain.lobes.features import Fbank
        transform = Fbank(sample_rate=16000, n_mels=cfg['model']['tts']['zipvoice']['feat_dim'])


    # -------------------------
    # training loop
    # -------------------------

    outdir = Path(cfg["training"]["out_dir"])
    if not (outdir / 'cfg' / Path(args.config).name).is_file():
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / 'cfg').mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.config, outdir / 'cfg')

    step = current_epoch * epoch_batch_length if epoch_batch_length is not None else max(current_step, 0)
    model.train()
    start_time = time.time()
    last_validation_step = None

    for epoch in range(cfg["training"]["epochs"]):
        losses = []
        if epoch < current_epoch:
            continue

        print(f"\nEpoch {epoch}", flush=True)

        physical_batch_index = -1
        train_iterator = iter(train_loader)
        accumulation_target_seconds = float(
            cfg['data']['train'].get('streaming_gradient_accumulation_target_seconds', 0.0)
        ) if train_is_streaming else 0.0
        if accumulation_target_seconds < 0.0:
            raise ValueError('streaming_gradient_accumulation_target_seconds must be non-negative.')

        while True:
            physical_batches = []
            physical_examples = 0
            physical_audio_seconds = 0.0
            while not physical_batches or (
                train_is_streaming
                and not accumulation_target_reached(
                    physical_examples, physical_audio_seconds, cfg
                )
            ):
                try:
                    next_batch = next(train_iterator)
                except StopIteration:
                    break
                physical_batch_index += 1
                if cfg['data']['train']['epoch_batch_limit'] > 0 and physical_batch_index >= cfg['data']['train']['epoch_batch_limit']:
                    break
                physical_batches.append(next_batch)
                physical_examples += int(next_batch['wav'].shape[0])
                physical_audio_seconds += (
                    next_batch['wav_lengths'].sum().item()
                    / cfg['data']['sampling_rate']
                )

            if not physical_batches:
                break
            batch_index = physical_batch_index
            if epoch_batch_length is not None and step < current_step:
                step += 1
                continue
            gradient_plan, skipped_texts = plan_gradient_microbatches(physical_batches, cfg)
            all_text_ids = [text for batch in physical_batches for text in batch['text']]
            all_wav_lengths = torch.cat([batch['wav_lengths'] for batch in physical_batches])
            if skipped_texts:
                skipped_summary = ', '.join(
                    f"physical={physical_index}:index={index}:tokens={token_count}"
                    for physical_index, index, token_count in skipped_texts
                )
                print(f"Skipping Emilia utterances above the text safety limit: {skipped_summary}", flush=True)
            if not gradient_plan:
                print("Skipping Emilia batch: every transcript exceeded the text safety limit.", flush=True)
                continue

            total_target_frames = sum(target_frame_count for _, _, target_frame_count in gradient_plan)

            if accumulation_target_seconds > 0.0:
                print(
                    f'Optimizer update audio: {physical_audio_seconds:.1f}s '
                    f'(target {accumulation_target_seconds:.1f}s, '
                    f'{physical_examples} utterances).',
                    flush=True,
                )

            torch.manual_seed(step)
            optimizer.zero_grad()
            batch_loss = 0.0
            for physical_batch, indices, target_frame_count in gradient_plan:
                wavs = physical_batch['wav']
                wav_lengths = physical_batch['wav_lengths']
                text_ids = physical_batch['text']
                accents = physical_batch['accent']
                micro_wav_lengths = wav_lengths[indices]
                micro_wavs = wavs[indices, : int(micro_wav_lengths.max())].to(device)
                micro_text_ids = [text_ids[index] for index in indices]
                micro_accents = [accents[index] for index in indices] if accents else []
                loss_weight = target_frame_count / total_target_frames
                micro_loss = backward_train_microbatch(
                    model,
                    linearvc_model,
                    transform,
                    cfg,
                    micro_wavs,
                    micro_wav_lengths,
                    micro_text_ids,
                    micro_accents,
                    device,
                    loss_weight,
                    scaler,
                )
                batch_loss += micro_loss * loss_weight

            losses.append(batch_loss)

            if cfg['training']['use_grad_scaler']:
                scaler.unscale_(optimizer)
            if not (list(model.parameters())[0].grad.isnan().any()):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['clip_grad_norm'])
                if cfg['training']['use_grad_scaler']:
                    scaler.step(optimizer)
                else:
                    optimizer.step()
            if cfg['training']['use_grad_scaler']:
                scaler.update()

            if step % cfg['training']['log_every'] == 0:
                current_time = time.time()
                print(
                    f"epoch {epoch} | batch {batch_index} | microbatches {len(gradient_plan)} | "
                    f"max tokens {max(map(len, all_text_ids))} | time elapsed {format_time(start_time, current_time)} | "
                    f"total step {step} | loss {batch_loss:.4f}",
                    flush=True,
                )
                wandb.log(
                    data = {
                        "train_loss": batch_loss,
                        "train_microbatches": len(gradient_plan),
                        "train_max_tokens": max(map(len, all_text_ids)),
                        "train_physical_batches": len(physical_batches),
                        "train_effective_examples": int(all_wav_lengths.numel()),
                        "train_audio_seconds": all_wav_lengths.sum().item() / cfg['data']['sampling_rate'],
                    },
                    step = step
                )

            step += 1
            data_cursor = None
            if train_is_streaming:
                data_cursor = physical_batches[-1].get("stream_cursor")
                if data_cursor is None:
                    raise RuntimeError("Emilia training batch is missing its resume cursor.")
                data_cursor = {**data_cursor, "epoch": epoch}

            if step % cfg["training"]["save_every_steps"] == 0 and step > 0:
                validate_every_steps = int(
                    cfg["training"].get("validate_every_steps", cfg["training"]["save_every_steps"])
                )
                if validate_every_steps < 1:
                    raise ValueError("training.validate_every_steps must be positive")
                if step % validate_every_steps != 0:
                    save_checkpoint(
                        model,
                        optimizer,
                        step,
                        outdir / f"ckpt_step_{step}.pt",
                        outdir,
                        cfg,
                        data_cursor=data_cursor,
                    )
                    continue
                validation_loss = validation_pass(
                    dev_loader,
                    fixed_audio_batch,
                    linearvc_model,
                    transform,
                    cfg,
                    model,
                    device,
                    step,
                    epoch,
                )
                wandb.log(
                    data = {
                        "validation_loss": validation_loss,
                    },
                    step = step
                )
                save_checkpoint(
                    model,
                    optimizer,
                    step,
                    outdir / f"ckpt_loss_{validation_loss:.2f}_step_{step}.pt",
                    outdir,
                    cfg,
                    data_cursor=data_cursor,
                )
                last_validation_step = step

        
        current_time = time.time()
        if not losses:
            print(
                f"epoch {epoch} | Emilia cursor was already at the end of this sweep; advancing.",
                flush=True,
            )
            continue
        print(f"epoch {epoch} | time elapsed {format_time(start_time, current_time)} | avg loss {(sum(losses) / len(losses)):.4f}")
        if (
            epoch % cfg["training"]["save_every_epochs"] == 0
            and step > 0
            and step != last_validation_step
        ):
            validation_loss = validation_pass(
                dev_loader,
                fixed_audio_batch,
                linearvc_model,
                transform,
                cfg,
                model,
                device,
                step,
                epoch,
            )
            wandb.log(
                data = {
                    "validation_loss": validation_loss,
                },
                step = step
            )
            save_checkpoint(
                model,
                optimizer,
                step,
                outdir / f"ckpt_loss_{validation_loss:.2f}_epoch_{epoch}.pt",
                outdir,
                cfg,
                data_cursor=data_cursor,
            )

        scheduler.step()


if __name__ == "__main__":
    main()
