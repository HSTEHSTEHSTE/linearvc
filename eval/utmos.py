import argparse
import utmosv2
import torchaudio
from pathlib import Path
from tqdm import tqdm

def check_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--converted_dir",
        type=Path,
        help="converted speech directory",
    )
    return parser.parse_args()

def main(args):
    print("Converted dir: ", args.converted_dir)
    out_wav_path = Path(args.converted_dir)

    extensions = ['wav', 'flac']
    out_wavs = []
    for extension in extensions:
        out_wavs += out_wav_path.rglob('*.' + extension)

    model = utmosv2.create_model(pretrained=True)
    scores = []

    for out_wav_path in tqdm(out_wavs):
        out_wav, sr = torchaudio.load(out_wav_path)
        scores.append(model.predict(data=out_wav, sr=sr).item())

    print("UTMOS v2 score: ", sum(scores) / len(scores))

if __name__ == "__main__":
    args = check_argv()
    main(args)