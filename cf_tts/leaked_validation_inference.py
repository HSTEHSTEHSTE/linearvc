"""Generate a deliberately information-leaked CF-TTS validation sample.

This is a diagnostic only. It intentionally leaks three target-side signals:
the target's first three seconds as prompt, its WavLM-frame duration, and a
content-to-WavLM transform estimated from the complete target waveform.
"""

import argparse
from pathlib import Path

import soundfile as sf
import torch

from linearvc import linearvc
from linearvc.cf_tts.dataset import TTSDataset, TTS_Collate
from linearvc.cf_tts.models.tts import ZipVoice
from linearvc.cf_tts.train import build_fixed_validation_audio_batch, prepare_validation_features
from linearvc.cf_tts.utils.checkpoints import load_checkpoint
from linearvc.cf_tts.utils.common import invert_normalized_input, load_config, load_content_projection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument(
        '--sampling-steps', type=int, default=None,
        help='Override the configured ODE steps; useful for a fast diagnostic smoke test.',
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    cfg = load_config(config_path)
    device = torch.device('cuda:0')
    if not torch.cuda.is_available():
        raise RuntimeError('This diagnostic requires CUDA.')

    print('Loading TTS checkpoint.', flush=True)
    model = ZipVoice(**cfg['model']['tts']['zipvoice']).to(device).eval()
    load_checkpoint(model, checkpoint_path, device)

    print('Loading WavLM and HiFi-GAN.', flush=True)
    wavlm = torch.hub.load('bshall/knn-vc', 'wavlm_large', trust_repo=True, progress=True, device=device)
    hifigan, _ = torch.hub.load(
        'bshall/knn-vc', 'hifigan_wavlm', trust_repo=True, prematched=True, progress=True, device=device
    )
    linearvc_model = linearvc.LinearVC(wavlm, hifigan, device)

    transform = load_content_projection(cfg['training']['content_factorization'], device)

    print('Loading fixed validation target.', flush=True)
    dev_set = TTSDataset(config_file_path=str(config_path), split='dev')
    fixed_batch = build_fixed_validation_audio_batch(dev_set, TTS_Collate(cfg), cfg)
    target_length = int(fixed_batch['wav_lengths'][0])
    target_wav = fixed_batch['wav'][:1, :target_length].to(device)
    prompt_samples = int(3.0 * cfg['data']['sampling_rate'])
    leaky_prompt_wav = target_wav[:, :prompt_samples]

    with torch.inference_mode():
        print('Extracting target and target-prompt features.', flush=True)
        target_content, raw_target_wavlm = prepare_validation_features(
            target_wav, cfg, linearvc_model, transform
        )
        prompt_content, _ = prepare_validation_features(
            leaky_prompt_wav, cfg, linearvc_model, transform
        )
        target_frames = int((target_length - 400) // 320 + 1)
        prompt_lens = torch.tensor([prompt_content.shape[1]], dtype=torch.int64, device=device)
        generated_lens = torch.tensor([target_frames], dtype=torch.int64, device=device)

        print(f'Sampling {target_frames} target-duration frames.', flush=True)
        sampling_steps = args.sampling_steps or int(cfg['training']['sampling_steps'])
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)
            generated = model.sample(
                tokens=[fixed_batch['text'][0]],
                prompt_tokens=[[]],
                prompt_features=prompt_content,
                prompt_features_lens=prompt_lens,
                duration='real',
                features_lens=generated_lens,
                num_step=sampling_steps,
            )[0]

        if cfg['training']['normalize_input']:
            generated = invert_normalized_input(generated)
            target_content = invert_normalized_input(target_content)
        generated = generated / cfg['training']['feature_scale']
        target_content = target_content / cfg['training']['feature_scale']

        # Deliberate leak: use every target frame, not the separate prompt, to
        # map generated content features back into raw WavLM/HiFi-GAN space.
        leaky_transform = (
            torch.linalg.pinv(target_content[0, :target_frames])
            @ raw_target_wavlm[0, :target_frames]
        )
        generated_raw_wavlm = generated @ leaky_transform
        audio = linearvc_model.hifigan(generated_raw_wavlm).reshape(-1).detach().cpu().numpy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, cfg['data']['sampling_rate'])
    print(f'OUTPUT={output_path}', flush=True)
    print(f'TARGET={fixed_batch["wav_path"][0]}', flush=True)
    print(f'TARGET_SECONDS={target_length / cfg["data"]["sampling_rate"]:.3f}', flush=True)
    print(f'GENERATED_SECONDS={len(audio) / cfg["data"]["sampling_rate"]:.3f}', flush=True)
    print(f'SAMPLING_STEPS={sampling_steps}', flush=True)
    print('LEAKS=target_prompt,target_duration,target_full_wavlm_transform', flush=True)


if __name__ == '__main__':
    main()
