# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Window indexing / split helpers shared by the HumanAction LeRobot readers."""

from __future__ import annotations

import pytest

from cosmos_framework.data.generator.action.datasets.base_dataset import (
    build_window_index,
    snapped_num_frames,
    split_episode_ids,
)


def _episodes(lengths: list[int]) -> list[int]:
    return [episode for episode, length in enumerate(lengths) for _ in range(length)]


@pytest.mark.L0
def test_dense_windows_match_the_legacy_sliding_rule() -> None:
    episode_ids = _episodes([200, 50, 145])
    windows = build_window_index(episode_ids, source_stride=2, chunk_length=72)
    # Legacy rule: starts in range(ep_start, max(ep_start, ep_end - stride * chunk), sample_stride).
    expected = [(s, 72) for s in range(0, 200 - 144)] + [(s, 72) for s in range(250, 395 - 144)]
    assert windows == expected  # the 50-row episode cannot host a 145-row window
    strided = build_window_index(episode_ids, source_stride=2, chunk_length=72, sample_stride=16)
    assert strided == [(s, 72) for s in range(0, 56, 16)] + [(s, 72) for s in range(250, 251, 16)]
    for start, steps in windows:
        rows = list(range(start, start + 2 * steps + 1, 2))
        assert len({episode_ids[r] for r in rows}) == 1


@pytest.mark.L0
def test_snapped_frames_follow_the_one_plus_4n_rule() -> None:
    assert snapped_num_frames(native_rows=300, source_stride=2, chunk_length=72) == 73  # capped at chunk + 1
    assert snapped_num_frames(native_rows=146, source_stride=2, chunk_length=72) == 73
    assert snapped_num_frames(native_rows=100, source_stride=2, chunk_length=72) == 49  # 50 frames -> 1 + 4*12
    assert snapped_num_frames(native_rows=10, source_stride=2, chunk_length=72) == 5
    assert snapped_num_frames(native_rows=9, source_stride=2, chunk_length=72) == 0  # 4 frames: too short


@pytest.mark.L0
def test_snap_to_subtask_gives_one_variable_length_window_per_subtask() -> None:
    episode_ids = _episodes([300, 60])
    # Episode 0: subtasks of 160 / 100 / 40 rows; episode 1: one 60-row subtask.
    subtask_ids = [0] * 160 + [1] * 100 + [2] * 40 + [3] * 60
    windows = build_window_index(
        episode_ids, source_stride=2, chunk_length=72, snap_to_subtask=True, subtask_ids=subtask_ids
    )
    assert windows == [(0, 72), (160, 48), (260, 16), (300, 28)]
    for start, steps in windows:
        rows = list(range(start, start + 2 * steps + 1, 2))
        assert len({subtask_ids[r] for r in rows}) == 1 and len({episode_ids[r] for r in rows}) == 1
    # A 5-row subtask is dropped, everything else unaffected.
    short = build_window_index(
        _episodes([300, 5]),
        source_stride=2,
        chunk_length=72,
        snap_to_subtask=True,
        subtask_ids=subtask_ids[:300] + [9] * 5,
    )
    assert short == windows[:3]
    with pytest.raises(ValueError):
        build_window_index(episode_ids, source_stride=2, chunk_length=72, snap_to_subtask=True)


@pytest.mark.L0
def test_split_is_deterministic_disjoint_and_complete() -> None:
    train = split_episode_ids(100, seed=7, val_ratio=0.1, split="train")
    val = split_episode_ids(100, seed=7, val_ratio=0.1, split="val")
    assert len(val) == 10 and len(train) == 90 and not set(train) & set(val)
    assert sorted(train + val) == list(range(100))
    assert val == split_episode_ids(100, seed=7, val_ratio=0.1, split="val")
    assert val != split_episode_ids(100, seed=8, val_ratio=0.1, split="val")
    assert sorted(split_episode_ids(100, seed=7, val_ratio=0.1, split="full")) == list(range(100))
    with pytest.raises(ValueError):
        split_episode_ids(100, seed=7, val_ratio=1.0, split="train")
    # keep_episodes restricts the window index to the chosen episodes.
    episode_ids = _episodes([150, 150, 150])
    kept = build_window_index(episode_ids, source_stride=2, chunk_length=72, keep_episodes={1})
    assert kept and all(150 <= start < 300 for start, _ in kept)
