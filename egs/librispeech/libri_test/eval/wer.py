import argparse
import jiwer
import json
from pathlib import Path
from whisper_normalizer.english import EnglishTextNormalizer

def check_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref_transcript_dir",
        type=Path,
        help="reference transcript directory", # /home/hltcoe/xli/ARTS/linearvc/exp/asr/LibriSpeech/orig
    )
    parser.add_argument(
        "--out_transcript_dir",
        type=Path,
        help="output transcript directory",
    )
    return parser.parse_args()

def main(args):
    english_normalizer = EnglishTextNormalizer()

    ref_transcript_dir = Path(args.ref_transcript_dir)
    out_transcript_dir = Path(args.out_transcript_dir)

    spks_long = out_transcript_dir.rglob('*.txt')
    spks = []
    for spk_long in spks_long:
        spks.append(spk_long.stem)

    # collect reference transcripts
    with open('linearvc/egs/librispeech/libri_test/speakers.json', 'r') as file:
        speakers = json.load(file)
    ref_transcripts = {}
    for speaker in speakers['lists']['test-clean_source']:
        with open(ref_transcript_dir / 'test-clean' / (str(speaker) + '.txt'), 'r') as file:
            for line in file:
                line_elements = line.strip().split('|')
                ref_transcripts[line_elements[0]] = english_normalizer(line_elements[1])

    wers = []
    for spk in spks:
        wers_spk = []
        out_transcripts = {}
        with open(out_transcript_dir / (spk + '.txt')) as out_transcript_file:
            for line in out_transcript_file:
                line = line.strip()
                line_elements = line.split('|')
                utt_name = line_elements[0].split('_')[0]
                out_transcript = english_normalizer(line_elements[1].strip())
                wer = jiwer.wer(ref_transcripts[utt_name], out_transcript)
                wers_spk.append(wer)
        wers += wers_spk

    wer = sum(wers) / len(wers)
    print("WER: ", wer)    

if __name__ == "__main__":
    args = check_argv()
    main(args)