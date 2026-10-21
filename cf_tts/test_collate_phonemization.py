"""CPU-only regression tests for malformed Emilia transcript handling."""

import unittest

import torch

from linearvc.cf_tts.dataset import TTS_Collate


class CollatePhonemizationTest(unittest.TestCase):
    def test_discards_only_unphonemizable_records_after_batch_fallback(self):
        collate = object.__new__(TTS_Collate)
        collate.tokenizer_type = "phone"
        collate.pre_phonemized = False
        collate.text_tokenizer = {"t": 0}
        collate.separator = None
        collate.use_accent = False
        collate.style_prompt_samples = 3
        collate._load_wav = lambda item: item["wav"]

        def phonemize(texts, **_kwargs):
            if len(texts) > 1:
                return ["t"]  # Force the per-transcript recovery path.
            return [] if texts[0] == "bad metadata" else ["t"]

        collate.phonemize = phonemize
        result = collate(
            [
                {"text": "good metadata", "speaker": "good", "wav": torch.ones(4)},
                {"text": "bad metadata", "speaker": "bad", "wav": torch.ones(6)},
            ]
        )

        self.assertEqual(result["text"], [[0]])
        self.assertEqual(result["speaker"], ["good"])
        self.assertEqual(result["wav"].shape, (1, 4))
        self.assertEqual(result["wav_lengths"].tolist(), [4])
        self.assertEqual(result["wav_path"], [None])


if __name__ == "__main__":
    unittest.main()
