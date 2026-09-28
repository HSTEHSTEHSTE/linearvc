import os
import argparse
import json
import random
from pathlib import Path
import torchaudio

def check_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--librispeech_root",
        type=Path,
    )
    parser.add_argument(
        "--out_wavs_path",
        type=Path,
    )
    parser.add_argument(
        "--dump_path",
        type=Path,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )
    parser.add_argument(
        "--set",
        type=str,
        help="librispeech, cv",
        default="librispeech"
    )
    parser.add_argument(
        "--num_stimuli",
        type=int,
        default=30
    )
    return parser.parse_args()

def main(args):
    out_wavs_path_root = Path(args.out_wavs_path)
    dump_path_root = Path(args.dump_path)
    librispeech_root = Path(args.librispeech_root)
    with open('linearvc/egs/librispeech/libri_test/speakers.json', 'r') as file:
        speakers = json.load(file)

    in_speakers = speakers['lists']['test-clean_source']
    out_speakers = speakers['lists']['test-other_target']
    maps = speakers['maps']

    random.seed(args.seed)

    with open('linearvc/egs/librispeech/libri_test/human_evals.json', 'r') as file:
        eval_maps = json.load(file)
    include_subsets = ['knnvc', 'seedvc', 'linearvc', 'cf/0/W0/75', 'cf/3/W1/75', 'cf_fl/10000/3/W1/75']
    with open('linearvc/egs/librispeech/libri_test/human_eval_filelist.json', 'r') as file:
        eval_filelist = json.load(file)[:args.num_stimuli]

    reference_books = {}
    for out_speaker in out_speakers:
        books = list((librispeech_root / "test-other" / out_speaker).iterdir())
        if len(books) == 1:
            reference_books[out_speaker] = {
                'refs': list(books[0].rglob("*.flac")),
            }
        else:
            books = random.sample(books, 2)
            reference_books[out_speaker] = {
                'refs': list(books[0].rglob("*.flac")),
                'tests': list(books[1].rglob("*.flac"))
            }
        random.shuffle(reference_books[out_speaker]['refs'])
        if 'tests' in reference_books[out_speaker]:
            random.shuffle(reference_books[out_speaker]['tests'])
    resamplers = {}
    test_files = {}
    ref_files = []
    for index, eval_file in enumerate(eval_filelist):
        dump_path = dump_path_root / 'test'
        target_speaker = eval_file.split('/')[1].split('_')[1].split('.')[0]
        target_file = dump_path / ('gt_' + str(index) + '.wav')
        target_file.parent.mkdir(parents=True, exist_ok=True)
        if 'tests' in reference_books[target_speaker]:
            test_file = reference_books[target_speaker]['tests'].pop()
        else:
            test_file = reference_books[target_speaker]['refs'].pop()
        wav, sr = torchaudio.load(str(test_file))
        if sr != 16000:
            if sr not in resamplers:
                resamplers[sr] = torchaudio.transforms.Resample(sr, 16000)
            wav = resamplers[sr](wav)
            sr = 16000
        test_files['gt_' + str(index) + '.wav'] = wav

        refs_path = dump_path_root / 'ref'
        ref_file_src = reference_books[target_speaker]['refs'].pop()
        ref_file = refs_path / ('gt_' + str(index) + '.wav')
        ref_file.parent.mkdir(parents=True, exist_ok=True)
        wav, sr = torchaudio.load(str(ref_file_src))
        if sr != 16000:
            if sr not in resamplers:
                resamplers[sr] = torchaudio.transforms.Resample(sr, 16000)
            wav = resamplers[sr](wav)
            sr = 16000
        ref_files.append(wav)

    for eval_set in include_subsets:
        out_wavs_path = out_wavs_path_root / eval_set
        dump_path = dump_path_root / 'test'
        for index, eval_file in enumerate(eval_filelist):
            src_file = out_wavs_path / eval_file
            target_file = dump_path / (eval_maps[eval_set] + '_' + str(index) + '.wav')
            target_file.parent.mkdir(parents=True, exist_ok=True)
            wav, sr = torchaudio.load(str(src_file))
            if sr != 16000:
                if sr not in resamplers:
                    resamplers[sr] = torchaudio.transforms.Resample(sr, 16000)
                wav = resamplers[sr](wav)
                sr = 16000
            test_files[(eval_maps[eval_set] + '_' + str(index) + '.wav')] = wav

    test_file_names = list(test_files.keys())
    random.shuffle(test_file_names)
    for index, test_file_name in enumerate(test_file_names):
        test_file_index = int(test_file_name.split('.')[0].split('_')[-1])
        ref_wav = ref_files[test_file_index]
        torchaudio.save(str(dump_path_root / 'test' / (str(index) + '_' + test_file_name)), test_files[test_file_name], 16000)
        torchaudio.save(str(dump_path_root / 'ref' / (str(index) + '_' + test_file_name)), ref_wav, 16000)


if __name__ == "__main__":
    args = check_argv()
    main(args)