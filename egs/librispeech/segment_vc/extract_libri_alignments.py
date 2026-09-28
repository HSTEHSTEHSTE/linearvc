import json
from tqdm import tqdm
from datasets import load_dataset

ds = load_dataset("gilkeyio/librispeech-alignments")
alignments = {}
for subset in ds:
    print(subset)
    for item in tqdm(ds[subset]):
        alignments[item['id']] = item['phonemes']

with open('/home/hltcoe/xli/ARTS/linearvc/exp/asr/LibriSpeech/alignments/alignments.json', 'w') as file:
    json.dump(alignments, file)