import argparse, json, random
import torch, torchaudio
import numpy as np
from pathlib import Path
from tqdm import tqdm
from linearvc.linearvc import LinearVC

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
        "--seed",
        type=int,
        default=41,
    )
    parser.add_argument(
        "--set_num",
        type=int,
    )
    parser.add_argument(
        "--W_type",
        type=str, # W0 (real S_dagger), W1 (UTXSS), W2 (ST), W3(S_rand_dagger)
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
    return parser.parse_args()


def main(args):
    print("Librispeech root: ", args.librispeech_root)
    print("Out dir: ", args.out_dir)
    print("Num utt per speaker: ", args.num_utt_per_speaker)
    librispeech_root = Path(args.librispeech_root)
    out_dir = Path(args.out_dir) / str(args.set_num)
    feat_path = Path(args.feat_path)

    random.seed(args.seed)

    with open('linearvc/egs/librispeech/libri_test/speakers.json', 'r') as file:
        speakers = json.load(file)
    
    in_speakers = speakers['lists']['test-clean_source']
    out_speakers = speakers['lists']['test-other_target']
    maps = speakers['maps']

    # Load all the required models
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
    linearvc_model = LinearVC(wavlm, hifigan, device)
    transforms_path = Path('linearvc/exp/interspeech/cf/transforms')
    ranks = [10, 20, 30, 50, 75, 100]

    if args.W_type == 'W3':
        assert args.set_num in [1, 3]
        W_speakers = speakers["anchors"][str(args.set_num)]
        print("W_speakers: ", W_speakers)

    for rank in tqdm(ranks):
        transforms = np.load(transforms_path / str(args.set_num) / ('rank_' + str(rank)) / 'transforms.npy', allow_pickle=True).item()
        for speaker in transforms:
            transforms[speaker] = torch.tensor(transforms[speaker]).to(device).float()
        if args.W_type == 'W0':
            assert args.set_num in [0, 2]
            W = None
        elif args.W_type == 'W1':
            assert args.set_num in [1, 3]
            W = np.load(transforms_path / str(args.set_num) / ('rank_' + str(rank)) / 'UTXSS.npy')
            W_tensor = torch.tensor(W).to(device).float()
        elif args.W_type == 'W2':
            assert args.set_num in [1, 3]
            W = np.load(transforms_path / str(args.set_num) / ('rank_' + str(rank)) / 'ST.npy')
            W_tensor = torch.tensor(W).to(device).float()
        elif args.W_type == 'W3':
            assert args.set_num in [1, 3]
            Ws = {}
            for W_speaker in W_speakers:
                Ws[W_speaker] = torch.tensor(torch.linalg.pinv(transforms[W_speaker])).to(device).float()

        if args.set_num in [2, 3]:
            transforms_target = {}
            for out_speaker in out_speakers:
                out_feats = np.load(feat_path / 'test-other' / (out_speaker + '.npy'))
                out_feats = torch.tensor(out_feats[:args.frame_limit]).to(device).float()
                if args.W_type in ['W1', 'W2']:
                    W_tensor = torch.tensor(W).to(device).float()
                    transform_content = torch.matmul(out_feats, W_tensor).squeeze(0)
                    transform_tgt = torch.matmul(torch.linalg.pinv(transform_content), out_feats.squeeze(1)) # [r, 1024]
                    transform_tgt = torch.tensor(transform_tgt).to(device)
                    transforms_target[out_speaker] = transform_tgt
                elif args.W_type == 'W3':
                    transforms_target[out_speaker] = {}
                    for W_name in Ws:
                        W = Ws[W_name]
                        transform_content = torch.matmul(out_feats, W).squeeze(0).float()
                        transform_tgt = torch.matmul(torch.linalg.pinv(transform_content), out_feats.squeeze(1)) # [r, 1024]
                        transform_tgt = torch.tensor(transform_tgt).to(device)
                        transforms_target[out_speaker][W_name] = transform_tgt


        for in_speaker in tqdm(in_speakers):
            (out_dir / args.W_type / str(rank) / in_speaker).mkdir(parents=True, exist_ok=True)
            spk_wavs = list((librispeech_root / 'test-clean' / in_speaker).rglob('*.flac'))
            if args.W_type == 'W0':
                W = torch.linalg.pinv(transforms[in_speaker])
            for spk_wav in spk_wavs[:args.num_utt_per_speaker]:
                input_features = linearvc_model.get_features(str(spk_wav))
                input_features = input_features

                for out_speaker in speakers['maps'][in_speaker]:
                    if args.W_type in ['W0', 'W1', 'W2']:
                        if args.set_num in [0, 1]:
                            out_features = torch.matmul(torch.matmul(input_features, W_tensor), transforms[out_speaker]).float()
                        elif args.set_num in [2, 3]:
                            out_features = torch.tensor(torch.matmul(torch.matmul(input_features, W_tensor), transforms_target[out_speaker])).float()
                        wav_hat = hifigan(out_features.unsqueeze(0)).squeeze(0).detach().cpu()
                        torchaudio.save(str(out_dir / args.W_type / str(rank) / in_speaker / (spk_wav.stem + '_' + out_speaker + '.wav')), wav_hat, 16000)
                    elif args.W_type == 'W3':
                        for W in Ws:
                            (out_dir / args.W_type / str(rank) / in_speaker / W).mkdir(parents=True, exist_ok=True)
                            if args.set_num in [0, 1]:
                                out_features = torch.matmul(torch.matmul(input_features, Ws[W]), transforms[out_speaker]).to(device).float()
                            else:
                                out_features = torch.tensor(torch.matmul(torch.matmul(input_features, Ws[W]), transforms_target[out_speaker][W])).to(device).float()
                            wav_hat = hifigan(out_features.unsqueeze(0)).squeeze(0).detach().cpu()
                            torchaudio.save(str(out_dir / args.W_type / str(rank) / in_speaker / W / (spk_wav.stem + '_' + out_speaker + '.wav')), wav_hat, 16000)


if __name__ == "__main__":
    args = check_argv()
    main(args)