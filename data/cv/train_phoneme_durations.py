from pathlib import Path
import re
from tqdm import tqdm
import pandas as pd
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
separator = Separator(phone='-', word=' ')
punctuation_marks = ''.join([";", ":", ",", ".", "!", "?", "¡", "¿", "—", "…", "\"", "«", "»", "“", "”", "(", ")", "{", "}", "[", "]"])
backend = EspeakBackend(
    language='en-us',
    preserve_punctuation=True,
    punctuation_marks=punctuation_marks,
    words_mismatch='ignore'
)
phonemize = backend.phonemize

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torchaudio
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1')
pipeline.to(device)

cv_root = Path('/home/hltcoe/xli/ARTS/geolocation/icefall/egs/radio/geolocation/corpora/commonvoice/en/clips')

transcript_df_file = Path('/home/hltcoe/xli/ARTS/linearvc/exp/asr/CommonVoice/clips/train.tsv')
target_df_file = Path('/home/hltcoe/xli/ARTS/linearvc/exp/asr/CommonVoice/clips/phonemes/train.tsv')
transcript_df = pd.read_csv(transcript_df_file, sep='|', header=0, index_col=None, quoting=3)
transcript_df['phones'] = ''
transcript_df['duration_vad'] = None
sentences = []
resamplers = {}
for index, entry in tqdm(transcript_df.iterrows(), total=transcript_df.shape[0]):
    sentence = str(entry['sentence'])
    assert len(sentence) > 0
    sentences.append(sentence)

    wav, sr = torchaudio.load(str(cv_root / entry['path']), backend='sox')
    if sr != 16000:
        if sr not in resamplers:
            resamplers[sr] = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        wav = resamplers[sr](wav)
        sr = 16000
    diarization = pipeline(
        {
            'waveform': wav,
            'sample_rate': sr,
        },
        num_speakers = 1
    )
    duration_vad = 0
    for segment, _, _ in diarization.itertracks(yield_label=True):
        duration_vad += segment.end - segment.start
    transcript_df.loc[index, 'duration_vad'] = duration_vad
phones = phonemize(sentences, separator=separator, strip=True)
phones = [re.sub(r"""([;:,.!?¡¿—…"«»“”\(\)\{\}\[\]])""", r"-\1", phone) for phone in phones]
phones = [phone.replace('- ', '-').replace(' ', '-').split('-') for phone in phones]
phones = ['_'.join(phone) for phone in phones]
transcript_df.loc[:, 'phones'] = phones

transcript_df.to_csv(target_df_file, sep='|', index=False)