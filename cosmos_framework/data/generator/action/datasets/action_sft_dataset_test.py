# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Streaming episode shuffler: every (rank, worker) shard yields, even when episodes are fewer than shards."""

from __future__ import annotations

import itertools

import pytest
from torch.utils.data import Dataset

from cosmos_framework.data.generator.action.datasets.action_sft_dataset import ActionIterableShuffleDataset


class _Windows(Dataset):
    """Stand-in for ActionSFTDataset: ``blocks`` = per-episode (start, length) over flat window indices."""

    def __init__(self, blocks: list[tuple[int, int]]):
        self._blocks = blocks
        self._n = sum(length for _, length in blocks)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> int:
        return idx

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        return self._blocks


def _one_epoch(stream: ActionIterableShuffleDataset, num_items: int) -> list[int]:
    return list(itertools.islice(iter(stream), num_items))


@pytest.mark.L0
def test_single_episode_is_sharded_by_window_across_ranks() -> None:
    """One episode of 10 windows, 4 ranks x 1 worker: episode-level sharding would starve 3 ranks."""
    windows = _Windows([(0, 10)])
    seen: list[int] = []
    for rank in range(4):
        stream = ActionIterableShuffleDataset(windows, seed=1)  # type: ignore[arg-type]
        stream.shard_world_size, stream.shard_rank = 4, rank
        # 10 windows over 4 shards -> shards get 3/3/2/2 windows per epoch; take one epoch's worth.
        per_epoch = len([i for i in range(10) if i % 4 == rank])
        got = _one_epoch(stream, per_epoch)
        assert got, f"rank {rank} yielded nothing"
        seen += got
    assert sorted(seen) == list(range(10))  # disjoint and complete over one epoch


@pytest.mark.L0
def test_many_episodes_keep_episode_level_sharding_and_sequential_windows() -> None:
    blocks = [(i * 5, 5) for i in range(8)]  # 8 episodes x 5 windows, 2 shards
    windows = _Windows(blocks)
    for rank in range(2):
        stream = ActionIterableShuffleDataset(windows, seed=3)  # type: ignore[arg-type]
        stream.shard_world_size, stream.shard_rank = 2, rank
        got = _one_epoch(stream, 20)  # 4 episodes x 5 windows per shard per epoch
        # Windows arrive in runs of 5 consecutive indices (sequential within an episode).
        for i in range(0, 20, 5):
            run = got[i : i + 5]
            assert run == list(range(run[0], run[0] + 5)) and run[0] % 5 == 0
