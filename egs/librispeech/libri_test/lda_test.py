import argparse, json, pickle
import torch, torchaudio
import numpy as np
from pathlib import Path
from tqdm import tqdm
from linearvc.randomized_lda import RandomizedLDA

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def check_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--librispeech_root",
        type=Path,
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        help="output speech directory",
    )
    parser.add_argument(
        "--num_utt_per_speaker",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--feat_path",
        type=Path,
    )
    parser.add_argument(
        "--frame_limit",
        type=int,
        default=500
    )
    parser.add_argument(
        "--lda_path",
        type=Path,
        help="pkl file"
    )
    return parser.parse_args()


def main(args):
    print("Librispeech root: ", args.librispeech_root)
    print("Out dir: ", args.out_dir)
    print("Num utt per speaker: ", args.num_utt_per_speaker)
    print("Feat path: ", args.feat_path)
    print("Frame limit: ", args.frame_limit)
    librispeech_root = Path(args.librispeech_root)
    out_dir = Path(args.out_dir)
    feat_path = Path(args.feat_path)
    lda_path = Path(args.lda_path)

    with open('linearvc/egs/librispeech/libri_test/speakers.json', 'r') as file:
        speakers = json.load(file)
    
    in_speakers = speakers['lists']['test-clean_source']
    out_speakers = speakers['lists']['test-other_target']

    # Load all the required models
    with open(lda_path, 'rb') as file:
        lda = pickle.load(file)

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

    transforms_target = {}
    uscf = torch.tensor(lda.scalings_).float().to(device)
    for out_speaker in out_speakers:
        out_feats = torch.tensor(np.load(feat_path / 'test-other' / (out_speaker + '.npy'))[:args.frame_limit]).to(device).float()
        transform_content = torch.matmul(out_feats, uscf)
        transform_tgt = torch.matmul(torch.linalg.pinv(transform_content), out_feats) # [r, 1024]
        transforms_target[out_speaker] = transform_tgt

    for in_speaker in tqdm(in_speakers):
        (out_dir / in_speaker).mkdir(parents=True, exist_ok=True)
        spk_wavs = list((librispeech_root / 'test-clean' / in_speaker).rglob('*.flac'))
        for out_speaker in speakers['maps'][in_speaker]:
            for spk_wav in spk_wavs[:args.num_utt_per_speaker]:
                wav, sr = torchaudio.load(str(spk_wav))
                with torch.inference_mode():
                    input_features, _ = wavlm.extract_features(wav.to(device), output_layer=6)
                new_feats = torch.matmul(torch.matmul(input_features.float(), uscf), transforms_target[out_speaker])
                wav_hat = hifigan(new_feats).squeeze(0).detach().cpu()
                torchaudio.save(str(out_dir / in_speaker / (spk_wav.stem + '_' + out_speaker + '.wav')), wav_hat, 16000)


if __name__ == "__main__":
    args = check_argv()
    main(args)