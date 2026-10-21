"""CPU-safe unit coverage for CF-TTS distributed data partitioning."""

import signal
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import Sampler

from linearvc.cf_tts.dataset import DistributedBatchSampler, EmiliaStreamingDataset
from linearvc.cf_tts.train import DistributedState, PreemptionRequest, distributed_barrier


class StaticBatchSampler(Sampler):
    def __init__(self, batches):
        self.batches = batches
        self.epochs = []

    def set_epoch(self, epoch):
        self.epochs.append(epoch)

    def __iter__(self):
        yield from self.batches

    def __len__(self):
        return len(self.batches)


class DistributedDataPartitionTest(unittest.TestCase):
    def test_preemption_request_records_sigusr1(self):
        request = PreemptionRequest()
        request.handle(signal.SIGUSR1, None)
        self.assertTrue(request.requested)

    def test_cuda_barrier_uses_the_rank_local_device(self):
        state = DistributedState(
            rank=2,
            local_rank=2,
            world_size=3,
            device=torch.device("cuda", 2),
            initialized=True,
        )
        with patch("linearvc.cf_tts.train.dist.barrier") as barrier:
            distributed_barrier(state)
        barrier.assert_called_once_with(device_ids=[2])

    def test_cpu_barrier_has_no_cuda_device_ids(self):
        state = DistributedState(
            rank=0,
            local_rank=0,
            world_size=2,
            device=torch.device("cpu"),
            initialized=True,
        )
        with patch("linearvc.cf_tts.train.dist.barrier") as barrier:
            distributed_barrier(state)
        barrier.assert_called_once_with()

    def test_length_aware_batches_are_disjoint_and_even_across_ranks(self):
        base = StaticBatchSampler([[index] for index in range(8)])
        samplers = [
            DistributedBatchSampler(base, num_replicas=3, rank=rank, drop_last=True)
            for rank in range(3)
        ]
        partitions = [list(sampler) for sampler in samplers]

        self.assertEqual([len(partition) for partition in partitions], [2, 2, 2])
        flattened = [sample for partition in partitions for batch in partition for sample in batch]
        self.assertEqual(sorted(flattened), list(range(6)))
        self.assertEqual(len(flattened), len(set(flattened)))

        samplers[0].set_epoch(7)
        self.assertEqual(base.epochs, [7])

    def test_emilia_shards_are_partitioned_across_rank_and_worker(self):
        shards = [f"Emilia/EN/{index}.tar" for index in range(12)]
        assigned = []
        for rank in range(3):
            dataset = object.__new__(EmiliaStreamingDataset)
            dataset.rank = rank
            dataset.world_size = 3
            dataset.shards = shards
            dataset.shuffle_seed = 42
            assigned.append(dataset._shards_for_worker(worker_id=0, num_workers=2))

        # This exercises one worker per rank. Together the partitions cover
        # exactly the rank/worker-strided subset and have no overlap.
        flattened = [shard for rank_shards in assigned for shard in rank_shards]
        expected = shards[0::2]  # worker 0 across three DDP ranks
        self.assertEqual(sorted(flattened), sorted(expected))
        self.assertEqual(len(flattened), len(set(flattened)))


if __name__ == "__main__":
    unittest.main()
