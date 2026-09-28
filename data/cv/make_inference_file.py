import pandas as pd
import random

test_file = '/home/hltcoe/xli/ARTS/geolocation/icefall/egs/radio/geolocation/corpora/test.tsv'
target_num_sentences = 100
test_df = pd.read_csv(test_file, sep='|', header=0, index_col=None, quoting=3)
random.seed(42)
rows_index = random.sample(list(range(test_df.shape[0])), target_num_sentences)
rows = test_df.iloc[rows_index]
rows = rows.drop(columns=['client_id', 'sentence_id', 'sentence_domain', 'up_votes', 'down_votes', 'age', 'gender', 'variant', 'locale', 'segment'])
rows['out_wav_name'] = rows['path'].apply(lambda x: x[:-4] + '.wav')
rows['path'] = rows['path'].apply(lambda x: '/home/hltcoe/xli/ARTS/geolocation/icefall/egs/radio/geolocation/corpora/commonvoice/en/clips/' + x)
rows = rows.rename(columns={'path': 'prompt_audio'})
rows = rows.rename(columns={'sentence': 'prompt_transcript'})
rows['target_speaker_audio'] = '/home/hltcoe/xli/ARTS/anon_baseline/data/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac'
rows['text'] = 'the mangle-knife john mangles inserting the blade of his poniard avoided the knife which now protruded above the soil but seized the hand that wielded it'
rows.to_csv('/home/hltcoe/xli/ARTS/linearvc/exp/tts/inference/inference_0.tsv', sep='|', index=False)