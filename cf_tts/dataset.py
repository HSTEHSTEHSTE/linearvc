# cf_tts/dataset.py
import os
import json, yaml
import random
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

import torch, torchaudio
from torch.utils.data import Dataset


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