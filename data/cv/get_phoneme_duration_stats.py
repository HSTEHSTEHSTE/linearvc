import json
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd

phonemes = ["\u0279", "a\u028a", "e\u026a", "\u0254\u0303", "\u0261", "x", "\u0254\u02d0", "\u026a\u02b2", "t", "\u0254", "n", "\u0254\u026a", "\u026c", "l", "u\u02d0", "\u028c", "i\u02d0", "\u026a\u0279", "b", "\u025b", "a\u026a", "p", "\u026a", "i\u02d0\u02b2", "f", "\u028a", "a\u026a\u025a", "\u0251\u02d0\u0279", "\u028a\u0279", "r", "\u014b", "\u00f0", "n\u02b2", "a\u026a\u0259", "t\u0283", "i\u0259", "m", "d", "\u0251\u02d0", "\u0294", "d\u0292", "w", "\u00e6", "i", "n\u0329", "o\u028a", "\u025c\u02d0", "\u0254\u02d0\u0279", "\u0292", "\u03b8", "\u027e", "\u0250", "\u025b\u0279", "\u1d7b", "k", "j", "i\u02b2", "\u0261\u02b2", "v", "s", "\u00e7", "\u0259", "\u0259l", "\u0283", "a\u026a\u02b2", "z", "h", "\u0251\u0303", "\u025a"]

phoneme_to_id = {}
for index, phoneme in enumerate(phonemes):
    phoneme_to_id[phoneme] = index

df_file = Path('/home/hltcoe/xli/ARTS/linearvc/exp/asr/CommonVoice/clips/phonemes/train.tsv')
df = pd.read_csv(df_file, sep='|', header=0, index_col=None, quoting=3)

As = []
Bs = []

for index, entry in tqdm(df.iterrows(), total=df.shape[0]):
    a = np.zeros(len(phonemes))
    phones = entry['phones'].split('_')
    for phone in phones:
        if phone in phonemes:
            a[phoneme_to_id[phone]] += 1
    b = entry['duration_vad']
    As.append(a)
    Bs.append(b)
As = np.stack(As, axis=0)
Bs = np.stack(Bs, axis=0)
solution = np.linalg.lstsq(As, Bs)
lengths = solution[0]
phoneme_to_avg_length = {}
for index, phoneme in enumerate(phonemes):
    phoneme_to_avg_length[phoneme] = lengths[index]
    print(phoneme, lengths[index])
with open('/home/hltcoe/xli/ARTS/linearvc/exp/asr/LibriSpeech/spm/phone_durations.json', 'w') as file:
    json.dump(phoneme_to_avg_length, file)