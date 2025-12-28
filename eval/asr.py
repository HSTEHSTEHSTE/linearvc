import argparse
import torch
import whisper
from pathlib import Path
from tqdm import tqdm

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def check_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--converted_dir",
        type=Path,
        help="converted speech directory",
    )
    parser.add_argument(
        "--out_transcript_dir",
        type=Path,
        help="directory to store ASR output",
    )
    return parser.parse_args()

def main(args):
    converted_dir = Path(args.converted_dir)
    out_transcript_dir = Path(args.out_transcript_dir)
    out_transcript_dir.mkdir(parents=True, exist_ok=True)

    extensions = ['wav', 'flac']
    spks_long = converted_dir.iterdir()
    spks = []
    for spk_long in spks_long:
        spks.append(str(spk_long).split('/')[-1])

    model = whisper.load_model('large', device="cuda")

    for spk in tqdm(spks):
        wavs = []
        transcripts = {}
        for extension in extensions:
            wavs += (converted_dir / spk).rglob('*.' + extension)

        for wav in tqdm(wavs):
            transcript = model.transcribe(str(wav), language="english")
            transcripts[wav.stem] = transcript['text']

        with open(out_transcript_dir / (spk + '.txt'), 'w') as out_transcript_file:
            for transcript in transcripts:
                out_transcript_file.write(transcript + '|' + transcripts[transcript] + '\n')


if __name__ == "__main__":
    args = check_argv()
    main(args)