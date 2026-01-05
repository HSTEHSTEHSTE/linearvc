# cf_tts/dataset.py
import os
import json, yaml
import random
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

import torch, torchaudio
from torch.utils.data import Dataset
from torch.utils.data import Sampler


class TTSDatum:
    """
    One training example:
      text → phonemes → acoustic features → speaker
    """
    def __init__(
        self,
        wav_path: str,
        text: str,
        speaker_id: str,
        num_frames: int,
    ):
        self.wav_path = wav_path
        self.text = text
        self.speaker_id = speaker_id
        self.num_frames = num_frames


class TTSDataset(Dataset):
    """
    Generic TTS dataset used by CF-TTS.

    Expected manifest format (jsonl or csv-like):
        wav_path | text | speaker_id(optional)

    Audio loading and feature extraction is kept here so
    training code stays clean, meow 🐾
    """

    def __init__(
        self,
        config_file_path: str,
    ):
        self.data: List[TTSDatum] = []

        # read config
        with open(config_file_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)

        # load data
        print("Reading Data")
        for subset in self.config['data']['librispeech_subsets']:
            print("Processing ", subset)
            transcript_files = list((Path(self.config['data']['librispeech_transcript_path']) / subset).glob('*.txt'))
            spks = []
            for transcript_file in tqdm(transcript_files):
                spk = transcript_file.stem
                spks.append(spk)
                with open(transcript_file, 'r') as file:
                    for line in file:
                        line = line.strip().split('|')
                        text = line[1].strip()
                        filename = line[0].strip()
                        filename_elements = filename.split('-')
                        wav_path = Path(self.config['data']['librispeech_audio_path']) / subset / spk / filename_elements[1] / (filename + '.flac')
                        num_frames = torchaudio.info(wav_path).num_frames
                        self.data.append(TTSDatum(
                            wav_path=wav_path,
                            text=text,
                            speaker_id=spk,
                            num_frames=num_frames
                        ))
        
        self.data.sort(key=lambda x: x.num_frames)

    # -------------------------
    # torch dataset
    # -------------------------

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        wav, sr = torchaudio.load(item.wav_path)

        wav = wav.squeeze(0)

        wav = wav[: self.config['data']['max_audio_len']]

        return {
            "wav": wav,
            "text": item.text,
            "speaker": item.speaker_id,
        }


# -------------------------
# sampler
# -------------------------
class FrameBatchSampler(Sampler):
    """
    Groups samples so that the total number of audio frames
    per batch does not exceed max_frames.

    This gives:
        short utterances → large batches
        long utterances  → small batches
    """

    def __init__(
        self,
        dataset,
        config_file_path: str,
    ):
        self.dataset = dataset

        # read config
        with open(config_file_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)
        self.max_frames = self.config['training']['max_frames_per_batch']

        self.indices = list(range(len(dataset)))

        self.batches = []
        batch = []
        total_frames = 0
        for idx in self.indices:
            frames = self.dataset.data[idx].num_frames

            # if this utterance alone is too big, force it into its own batch
            if frames > self.max_frames:
                if batch:
                    self.batches.append(batch)
                    batch = []
                    total_frames = 0
                self.batches.append([idx])
                continue

            if total_frames + frames > self.max_frames and batch:
                self.batches.append(batch)
                batch = []
                total_frames = 0

            batch.append(idx)
            total_frames += frames

        if batch:
            self.batches.append(batch)
        if self.config['training']['shuffle_batches']:
            random.seed(self.config['training']['random_seed'])
            random.shuffle(self.batches)
        self.batches = iter(self.batches)


    def __iter__(self):
        yield next(self.batches)
        

    def __len__(self):
        # PyTorch allows this to be approximate
        return len(self.indices)


# -------------------------
# batching
# -------------------------

def tts_collate(batch: List[Dict]):
    """
    Pads variable-length audio and text for batching.
    Text tokenization happens in the model.
    """
    wavs = [b["wav"] for b in batch]
    speakers = [b["speaker"] for b in batch]
    texts = [b["text"] for b in batch]

    lengths = torch.tensor([w.size(0) for w in wavs])
    max_len = max(lengths)

    padded = torch.zeros(len(wavs), max_len)
    for i, w in enumerate(wavs):
        padded[i, : w.size(0)] = w

    return {
        "wav": padded,
        "wav_lengths": lengths,
        "text": texts,
        "speaker": speakers,
    }

if __name__ == "__main__":
    # test dataset object
    dataset = TTSDataset(config_file_path='cf_tts/config/config.yaml')
    print(len(dataset))
    print(dataset.__getitem__(0))

    # test sampler
    sampler = FrameBatchSampler(dataset, config_file_path='cf_tts/config/config.yaml')
    print(len(sampler))
    for i in range(5):
        print(next(iter(sampler)))