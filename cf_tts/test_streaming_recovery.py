"""Focused local tests for Emilia retry and checkpoint-cursor recovery."""

import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from linearvc.cf_tts.dataset import EmiliaLengthSortedBatchDataset, EmiliaStreamingDataset
from linearvc.cf_tts.train import accumulation_target_reached, plan_gradient_microbatches, split_text_microbatches
from linearvc.cf_tts.utils.checkpoints import load_checkpoint, save_checkpoint
from linearvc.cf_tts.utils.common import load_content_projection


class EmiliaStreamingRecoveryTest(unittest.TestCase):
    def test_short_emilia_clips_expand_linearly_and_updates_target_audio_seconds(self):
        dataset = object.__new__(EmiliaLengthSortedBatchDataset)
        dataset.reference_batch_size = 24
        dataset.reference_audio_samples = 30 * 16000
        dataset.max_batch_size = 128
        self.assertEqual(dataset._batch_capacity(30 * 16000), 24)
        self.assertEqual(dataset._batch_capacity(10 * 16000), 72)
        self.assertEqual(dataset._batch_capacity(3 * 16000), 128)

        cfg = {
            "data": {
                "train": {
                    "streaming_gradient_accumulation_target_seconds": 4000,
                    "streaming_gradient_accumulation_target_examples": 24,
                }
            }
        }
        self.assertFalse(accumulation_target_reached(1000, 3999.9, cfg))
        self.assertTrue(accumulation_target_reached(1, 4000.0, cfg))

    def test_content_projection_defaults_to_utxss_and_keeps_legacy_toggle(self):
        anchor_transform = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)
        utxss = np.arange(6, dtype=np.float32).reshape(3, 2)
        with tempfile.TemporaryDirectory() as directory:
            factorization_path = Path(directory) / "transforms.npy"
            np.save(factorization_path, {"103": anchor_transform})
            np.save(Path(directory) / "UTXSS.npy", utxss)
            factorization = {
                "type": "content",
                "content_factorization_file": str(factorization_path),
            }

            default_projection = load_content_projection(factorization, torch.device("cpu"))
            self.assertTrue(torch.equal(default_projection, torch.from_numpy(utxss)))

            factorization["content_projection"] = "pinv_anchor"
            legacy_projection = load_content_projection(factorization, torch.device("cpu"))
            self.assertTrue(
                torch.allclose(
                    legacy_projection, torch.from_numpy(np.linalg.pinv(anchor_transform).astype(np.float32))
                )
            )

    def test_text_microbatches_preserve_effective_batch_and_bound_attention_cost(self):
        tokens = [[1] * 800 for _ in range(12)] + [[1] * 1601]
        microbatches, skipped = split_text_microbatches(
            tokens,
            token_square_budget=24 * 400 * 400,
            max_tokens_per_utterance=1600,
        )

        self.assertEqual(microbatches, [list(range(6)), list(range(6, 12))])
        self.assertEqual(skipped, [(12, 1601)])
        for indices in microbatches:
            self.assertLessEqual(sum(len(tokens[index]) ** 2 for index in indices), 24 * 400 * 400)

    def test_gradient_accumulation_keeps_physical_batches_separate(self):
        cfg = {
            "training": {"content_factorization": {"type": "content"}},
            "model": {"tts": {"zipvoice": {"style_prompt_frames": 150}}},
            "data": {
                "train": {
                    "streaming_token_square_budget": 24 * 400 * 400,
                    "streaming_max_tokens_per_utterance": 1600,
                }
            },
        }
        first = {
            "wav_lengths": torch.tensor([160000, 160000]),
            "text": [[1] * 10, [1] * 10],
        }
        second = {
            "wav_lengths": torch.tensor([160000, 160000]),
            "text": [[1] * 10, [1] * 10],
        }

        plan, skipped = plan_gradient_microbatches([first, second], cfg)

        self.assertEqual(skipped, [])
        self.assertEqual(len(plan), 2)
        self.assertIs(plan[0][0], first)
        self.assertIs(plan[1][0], second)
        self.assertEqual(plan[0][1], [0, 1])
        self.assertEqual(plan[1][1], [0, 1])

    def test_retry_reopens_shard_without_replaying_raw_samples(self):
        dataset = object.__new__(EmiliaStreamingDataset)
        dataset.shard_max_retries = 1
        dataset.shard_retry_backoff_seconds = 0.0
        dataset.shard_retry_max_backoff_seconds = 0.0
        dataset._prepare_sample = lambda sample: {"id": sample["id"]}

        attempts = []

        def open_stream(_path):
            attempts.append(1)
            if len(attempts) == 1:
                def broken_stream():
                    yield {"id": "a"}
                    raise tarfile.ReadError("unexpected end of data")
                return broken_stream()
            return iter([{"id": "a"}, {"id": "b"}, {"id": "c"}])

        dataset._open_shard_stream = open_stream
        samples = list(dataset._iter_shard("Emilia/EN/test.tar", shard_index=4))

        self.assertEqual([sample["id"] for sample in samples], ["a", "b", "c"])
        self.assertEqual([sample["_raw_sample_index"] for sample in samples], [0, 1, 2])
        self.assertEqual(len(attempts), 2)

    def test_completed_staged_shard_is_removed_after_consumption(self):
        dataset = object.__new__(EmiliaStreamingDataset)
        dataset.shard_max_retries = 0
        dataset.shard_retry_backoff_seconds = 0.0
        dataset.shard_retry_max_backoff_seconds = 0.0
        dataset._prepare_sample = lambda sample: {"id": sample["id"]}

        with tempfile.TemporaryDirectory() as directory:
            staged_path = Path(directory) / "Emilia" / "EN" / "test.tar"
            staged_path.parent.mkdir(parents=True)
            staged_path.write_bytes(b"complete local shard")
            dataset._open_shard_stream = lambda _path: (iter([{"id": "a"}]), staged_path)

            samples = list(dataset._iter_shard("Emilia/EN/test.tar", shard_index=0))

            self.assertEqual([sample["id"] for sample in samples], ["a"])
            self.assertFalse(staged_path.exists())

    def test_batch_cursor_resumes_at_next_batch(self):
        dataset = object.__new__(EmiliaLengthSortedBatchDataset)
        dataset.reference_audio_samples = 100
        dataset.reference_batch_size = 2
        dataset.max_batch_size = 2
        dataset.shuffle_seed = 42

        def samples():
            return [
                {"id": "a", "num_samples": 60, "audio_bytes": b"a", "_shard_path": "Emilia/EN/test.tar"},
                {"id": "b", "num_samples": 60, "audio_bytes": b"b", "_shard_path": "Emilia/EN/test.tar"},
                {"id": "c", "num_samples": 60, "audio_bytes": b"c", "_shard_path": "Emilia/EN/test.tar"},
                {"id": "d", "num_samples": 60, "audio_bytes": b"d", "_shard_path": "Emilia/EN/test.tar"},
            ]

        all_batches = list(dataset._batches_from_shard(samples(), shard_index=3))
        cursor = all_batches[0][0]["_stream_cursor"]
        resumed_batches = list(
            dataset._batches_from_shard(
                samples(), shard_index=3, first_batch_index=cursor["batch_index"]
            )
        )

        expected = [sample["id"] for batch in all_batches[1:] for sample in batch]
        actual = [sample["id"] for batch in resumed_batches for sample in batch]
        self.assertEqual(actual, expected)

    def test_checkpoint_persists_cursor_atomically(self):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        cursor = {
            "epoch": 0,
            "shard_index": 3,
            "batch_index": 8,
            "shard_path": "Emilia/EN/test.tar",
        }
        cfg = {
            "training": {
                "num_ckpt_best_loss": 1,
                "num_ckpt_latest_epochs": 1,
                "num_ckpt_latest_steps": 1,
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            outdir = Path(directory)
            path = outdir / "ckpt_loss_0.10_step_12.pt"
            save_checkpoint(model, optimizer, 12, path, outdir, cfg, data_cursor=cursor)
            loaded = load_checkpoint(
                torch.nn.Linear(2, 2), path, torch.device("cpu"), return_checkpoint=True
            )
            self.assertEqual(loaded["step"], 12)
            self.assertEqual(loaded["data_cursor"], cursor)
            self.assertFalse(path.with_suffix(".pt.tmp").exists())


if __name__ == "__main__":
    unittest.main()
