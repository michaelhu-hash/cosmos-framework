# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Abstract base class for Action LeRobot datasets."""

from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from cosmos_framework.data.generator.action.action_normalization import (
    denormalize_action,
    load_action_stats,
    normalize_action,
)
from cosmos_framework.data.generator.action.utils.action_processing import (
    ActionInitialStateNormalization,
    ActionNormalizer,
    load_action_normalizer,
)
from cosmos_framework.data.generator.action.utils.action_spec import ActionSpec
from cosmos_framework.data.generator.action.utils.domain_utils import get_domain_id
from cosmos_framework.data.generator.action.utils.pose_utils import compute_idle_frames

_MODE_CHOICES = ("forward_dynamics", "inverse_dynamics", "wam")
# Methods served by ``action_normalization.normalize_action`` (top-level stats JSON, optional clamp). Anything
# else is resolved through ``utils.action_processing.load_action_normalizer`` (nested ``global`` /
# ``global_raw`` stats blocks, e.g. ``piecewise_asinh_rot`` used by Cosmos3-Nano-HumanAction).
LEGACY_NORMALIZATION_METHODS = ("quantile", "meanstd", "minmax")
ANCHORED_POSE_CONVENTIONS = ("backward_anchored", "backward_chunk_anchored_8f", "backward_chunk_anchored_16f")
SUPPORTED_POSE_CONVENTIONS = ("backward_framewise", *ANCHORED_POSE_CONVENTIONS)
# Image2Action: ``initial_state="predict"`` prepends the frame-0 initial-state row a0 to inverse-dynamics /
# wam windows (``T + 1`` rows) and flags the item ``has_initial_state`` so the transform noises + supervises
# row 0 and the normalizer routes it to dedicated a0 stats. Forward dynamics never carries a0.
INITIAL_STATE_MODES = ("predict",)
INITIAL_STATE_ACTION_MODES = ("inverse_dynamics", "wam")
SPLIT_CHOICES = ("full", "train", "val")
# Snapped windows must be 1 + 4N video frames (tokenizer temporal compression) and at least 5 frames.
_SNAP_FRAME_GROUP = 4
_SNAP_MIN_FRAMES = 5


def split_episode_ids(total_episodes: int, seed: int, val_ratio: float, split: str) -> list[int]:
    """Deterministic episode POSITIONS for a train / val / full split (same rule as the DROID readers).

    ``round(total * val_ratio)`` episodes of a seeded permutation form the val split; the rest train.
    """
    if split not in SPLIT_CHOICES:
        raise ValueError(f"split must be one of {SPLIT_CHOICES}, got {split!r}")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    num_val = int(round(total_episodes * val_ratio))
    order = torch.randperm(total_episodes, generator=torch.Generator().manual_seed(seed)).tolist()
    if split == "train":
        return order[num_val:]
    if split == "val":
        return order[:num_val]
    return order


def snapped_num_frames(native_rows: int, source_stride: int, chunk_length: int) -> int:
    """Video frames of a subtask-snapped window: ``min(rows // stride, chunk_length + 1)`` rounded down to ``1 + 4N``;
    0 when the subtask is shorter than 5 frames (mirrors the internal ``HandPoseDataset`` snap rule)."""
    frames = min(native_rows // source_stride, chunk_length + 1)
    if frames < _SNAP_MIN_FRAMES:
        return 0
    return 1 + _SNAP_FRAME_GROUP * ((frames - 1) // _SNAP_FRAME_GROUP)


def build_window_index(
    episode_ids: Sequence[int],
    *,
    source_stride: int,
    chunk_length: int,
    sample_stride: int = 1,
    keep_episodes: Collection[int] | None = None,
    snap_to_subtask: bool = False,
    subtask_ids: Sequence[int] | None = None,
) -> list[tuple[int, int]]:
    """``(start_row, num_steps)`` windows over a flat, index-sorted row table whose episodes are contiguous.

    Dense (default): every start (step ``sample_stride``) whose ``source_stride * chunk_length + 1`` rows stay
    inside the episode; every window has ``chunk_length`` steps.
    ``snap_to_subtask``: ONE window per subtask (uniform sampling over subtasks instead of a bias toward long
    ones), starting at the subtask's first row, ``snapped_num_frames(...) - 1`` steps long, i.e. variable length
    up to ``chunk_length``; subtasks shorter than 5 frames are dropped. ``subtask_ids`` is the per-row
    ``subtask_index`` column. ``keep_episodes`` restricts to a split.
    """
    if snap_to_subtask and subtask_ids is None:
        raise ValueError("snap_to_subtask needs the per-row subtask_ids column")
    windows: list[tuple[int, int]] = []
    num_rows = len(episode_ids)
    episode_start = 0
    while episode_start < num_rows:
        episode = int(episode_ids[episode_start])
        episode_end = episode_start + 1
        while episode_end < num_rows and int(episode_ids[episode_end]) == episode:
            episode_end += 1
        if keep_episodes is None or episode in keep_episodes:
            if not snap_to_subtask:
                required_source_steps = source_stride * chunk_length
                windows.extend(
                    (start, chunk_length)
                    for start in range(
                        episode_start, max(episode_start, episode_end - required_source_steps), sample_stride
                    )
                )
            else:
                assert subtask_ids is not None
                starts = [episode_start] + [
                    row for row in range(episode_start + 1, episode_end) if subtask_ids[row] != subtask_ids[row - 1]
                ]
                for i, row_start in enumerate(starts):
                    native_rows = (starts[i + 1] if i + 1 < len(starts) else episode_end) - row_start
                    frames = snapped_num_frames(native_rows, source_stride, chunk_length)
                    if frames > 0:
                        windows.append((row_start, frames - 1))
        episode_start = episode_end
    return windows


class ActionBaseDataset(ABC, Dataset):
    """Abstract base for Action LeRobot datasets.

    Subclasses must implement the abstract methods listed below.
    """

    def __init__(
        self,
        root: str,
        domain_name: str,
        fps: float,
        chunk_length: int,
        mode: str,
        pose_convention: str,
        tolerance_s: float,
        viewpoint: str,
        action_normalization: str | None = "quantile",
        sample_stride: int = 1,
        stats_path: str | Path | None = None,
        initial_state: str | None = None,
        initial_state_stats_path: str | Path | None = None,
        split: str = "full",
        val_ratio: float = 0.0,
        split_seed: int = 42,
        snap_to_subtask: bool = False,
    ) -> None:
        super().__init__()
        if pose_convention not in SUPPORTED_POSE_CONVENTIONS:
            raise NotImplementedError(
                f"{type(self).__name__} supports pose conventions {SUPPORTED_POSE_CONVENTIONS}, got {pose_convention!r}."
            )

        self._fps = float(fps)
        self._dt = 1.0 / self._fps
        self._chunk_length = int(chunk_length)
        self._sample_stride = int(sample_stride)
        if self._sample_stride < 1:
            raise ValueError(f"sample_stride must be >= 1, got {self._sample_stride}")
        self._mode = mode
        self._pose_convention = pose_convention
        self._tolerance_s = float(tolerance_s)
        self._viewpoint = viewpoint
        self._domain_name = domain_name
        self._domain_id = get_domain_id(domain_name)
        self._action_normalization = action_normalization
        self._norm_stats: dict[str, torch.Tensor] | None = None
        # Optional per-instance stats file. One reader can serve several checkpoints whose normalizers differ
        # (e.g. base Cosmos3-Nano vs Cosmos3-Nano-HumanAction on the same hand-pose LeRobot data).
        self._stats_path_override: Path | None = Path(stats_path) if stats_path is not None else None
        self._action_normalizer: ActionNormalizer | None = None
        if initial_state is not None and initial_state not in INITIAL_STATE_MODES:
            raise ValueError(f"initial_state must be one of {INITIAL_STATE_MODES} or None, got {initial_state!r}")
        if initial_state is not None and action_normalization in LEGACY_NORMALIZATION_METHODS:
            raise NotImplementedError(
                "initial_state needs an asinh-family action normalizer with dedicated a0 stats; the legacy "
                f"{action_normalization!r} normalizer has no initial-state statistics"
            )
        self._initial_state = initial_state
        self._initial_state_stats_path_override: Path | None = (
            Path(initial_state_stats_path) if initial_state_stats_path is not None else None
        )
        self._initial_state_normalizer: ActionNormalizer | None = None
        if split not in SPLIT_CHOICES:
            raise ValueError(f"split must be one of {SPLIT_CHOICES}, got {split!r}")
        if split == "val" and val_ratio <= 0.0:
            raise ValueError("split='val' needs val_ratio > 0")
        self._split = split
        self._val_ratio = float(val_ratio)
        self._split_seed = int(split_seed)
        self._snap_to_subtask = bool(snap_to_subtask)
        # Set by _init_window_index(): (start_row, num_steps) per window + the episode of each window.
        self._windows: list[tuple[int, int]] | None = None
        self._window_episode_ids: list[int] = []
        self._source_stride: int = 1

        self._root = Path(root)
        self._info = json.loads((self._root / "meta" / "info.json").read_text())
        self._episodes = {
            int(row["episode_index"]): row
            for path in sorted((self._root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
            for row in pq.read_table(path).to_pylist()
        }
        tasks_df = pd.read_parquet(self._root / "meta" / "tasks.parquet")
        # LeRobot v2.x stores task text in a "task" column; v3.0 stores it as the
        # (unnamed) DataFrame index and keeps only "task_index" as a column.
        task_texts = tasks_df["task"] if "task" in tasks_df.columns else tasks_df.index
        self._tasks = {int(task_index): str(task) for task, task_index in zip(task_texts, tasks_df["task_index"])}
        # ``self._rows`` (the flat, index-sorted list of every frame dict) is built
        # lazily on first access — see the ``_rows`` property. Materializing all
        # ~18M frames as Python dicts plus a full sort costs ~13 min and tens of GB;
        # subclasses that build their own compact index (e.g. DROIDLeRobotDataset)
        # never touch it, so they must not pay for it at construction.
        self._rows_cache: list[dict[str, Any]] | None = None

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def chunk_length(self) -> int:
        return self._chunk_length

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        self._mode = value

    @property
    def domain_name(self) -> str:
        return self._domain_name

    @property
    def viewpoint(self) -> str:
        return self._viewpoint

    @property
    def domain_id(self) -> int:
        return self._domain_id

    @property
    def split(self) -> str:
        return self._split

    @property
    def snap_to_subtask(self) -> bool:
        return self._snap_to_subtask

    def _init_window_index(self, *, source_stride: int) -> None:
        """Build the window index over ``self._rows`` from the split / snap settings (call once from ``__init__``)."""
        rows = self._rows
        episode_ids = [int(row["episode_index"]) for row in rows]
        keep: set[int] | None = None
        if self._split != "full" or self._val_ratio > 0.0:
            unique = sorted(set(episode_ids))
            keep = {
                unique[pos] for pos in split_episode_ids(len(unique), self._split_seed, self._val_ratio, self._split)
            }
        subtask_ids = [int(row.get("subtask_index", -1)) for row in rows] if self._snap_to_subtask else None
        self._source_stride = int(source_stride)
        self._windows = build_window_index(
            episode_ids,
            source_stride=self._source_stride,
            chunk_length=self._chunk_length,
            sample_stride=self._sample_stride,
            keep_episodes=keep,
            snap_to_subtask=self._snap_to_subtask,
            subtask_ids=subtask_ids,
        )
        self._window_episode_ids = [episode_ids[start] for start, _ in self._windows]

    def _window_rows(self, idx: int) -> list[dict[str, Any]]:
        """Rows of window ``idx`` at the target fps: ``num_steps + 1`` frames, never crossing an episode."""
        if self._windows is None:
            raise RuntimeError(f"{type(self).__name__} did not call _init_window_index()")
        start, num_steps = self._windows[int(idx)]
        rows = self._rows[start : start + self._source_stride * num_steps + 1 : self._source_stride]
        if len(rows) != num_steps + 1:
            raise IndexError(f"Incomplete window at index {idx}.")
        episode_index = int(rows[0]["episode_index"])
        if any(int(row["episode_index"]) != episode_index for row in rows):
            raise IndexError(f"Window at index {idx} crosses an episode boundary.")
        return rows

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        """Per-episode ``(first_window, count)`` blocks for ``ActionIterableShuffleDataset``."""
        blocks: list[tuple[int, int]] = []
        for i, episode in enumerate(self._window_episode_ids):
            if blocks and self._window_episode_ids[blocks[-1][0]] == episode:
                blocks[-1] = (blocks[-1][0], blocks[-1][1] + 1)
            else:
                blocks.append((i, 1))
        return blocks

    @property
    def initial_state(self) -> str | None:
        """``"predict"`` when windows of the action-predicting modes carry the frame-0 row a0, else None."""
        return self._initial_state

    def wants_initial_state(self, mode: str) -> bool:
        """a0 is an OUTPUT of inverse dynamics / wam; forward dynamics never carries it."""
        return self._initial_state is not None and mode in INITIAL_STATE_ACTION_MODES

    @property
    def action_normalization(self) -> str:
        return self._action_normalization

    @property
    @abstractmethod
    def action_dim(self) -> int: ...

    @abstractmethod
    def _action_spec(self) -> ActionSpec: ...

    @property
    def action_names(self) -> list[str]:
        return self._action_spec().names

    @classmethod
    @abstractmethod
    def _stats_path(cls) -> Path:
        """Return the path to the stats JSON file for this dataset."""
        ...

    @classmethod
    def load_action_stats(cls) -> dict[str, torch.Tensor]:
        """Return action normalization stats for this dataset as torch tensors."""
        return {
            key: torch.from_numpy(value).float() for key, value in load_action_stats(str(cls._stats_path())).items()
        }

    def stats_path(self) -> Path:
        """Stats file used by this instance: the constructor override, else the class default."""
        return self._stats_path_override if self._stats_path_override is not None else type(self)._stats_path()

    @classmethod
    def _initial_state_stats_path(cls) -> Path:
        raise NotImplementedError(f"{cls.__name__} ships no initial-state (a0) stats; pass initial_state_stats_path")

    def initial_state_stats_path(self) -> Path:
        """a0 stats file used by this instance: the constructor override, else the class default."""
        if self._initial_state_stats_path_override is not None:
            return self._initial_state_stats_path_override
        return type(self)._initial_state_stats_path()

    def _resolve_initial_state_normalizer(self) -> ActionNormalizer:
        if self._initial_state_normalizer is None:
            self._initial_state_normalizer = load_action_normalizer(
                self._action_normalization,  # type: ignore[arg-type]
                stats_path=self.initial_state_stats_path(),
                apply_forward_clamp=False,
                expected_dim=self.action_dim,
            )
        return self._initial_state_normalizer

    def _has_initial_state_rows(self, action: torch.Tensor, has_initial_state: bool | None) -> bool:
        """Explicit flag wins; otherwise an a0-enabled dataset treats ``chunk_length + 1`` rows as a0 + deltas."""
        if has_initial_state is None:
            return self._initial_state is not None and action.shape[-2] == self._chunk_length + 1
        if has_initial_state and self._initial_state is None:
            raise ValueError(f"{type(self).__name__} was built without initial_state; it has no a0 normalizer")
        return has_initial_state

    def _normalizer_for(self, action: torch.Tensor, has_initial_state: bool | None) -> ActionNormalizer:
        base = self._resolve_action_normalizer()
        if not self._has_initial_state_rows(action, has_initial_state):
            return base
        return ActionInitialStateNormalization(base=base, initial_state=self._resolve_initial_state_normalizer())

    def _resolve_action_normalizer(self) -> ActionNormalizer:
        if self._action_normalizer is None:
            self._action_normalizer = load_action_normalizer(
                self._action_normalization,  # type: ignore[arg-type]
                stats_path=self.stats_path(),
                apply_forward_clamp=False,
                expected_dim=self.action_dim,
            )
        return self._action_normalizer

    def normalize(self, action: torch.Tensor, *, has_initial_state: bool | None = None) -> torch.Tensor:
        """Raw action -> model space, using this dataset's normalization method and stats.

        ``has_initial_state`` marks a ``[..., T+1, D]`` Image2Action window whose row 0 is the a0 row (own stats);
        None infers it from the row count for a0-enabled datasets.
        """
        if self._action_normalization is None:
            return action
        if self._action_normalization in LEGACY_NORMALIZATION_METHODS:
            return normalize_action(action, self._action_normalization, self._load_norm_stats())
        return self._normalizer_for(action, has_initial_state).normalize_action(action)

    def denormalize(self, action: torch.Tensor, *, has_initial_state: bool | None = None) -> torch.Tensor:
        """Model-space action -> raw action (inverse of :meth:`normalize`; same ``has_initial_state`` rule)."""
        if self._action_normalization is None:
            return action
        if self._action_normalization in LEGACY_NORMALIZATION_METHODS:
            return denormalize_action(action, self._action_normalization, self._load_norm_stats())
        return self._normalizer_for(action, has_initial_state).denormalize_action(action)

    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, Any]: ...

    def _compute_idle_frames(self, action: torch.Tensor) -> int:
        # Idle detection is defined on framewise one-step deltas. The anchored recipes (Cosmos3-Nano-HumanAction)
        # were trained without idle-frame annotations, so report none rather than misread anchored rows.
        if self._pose_convention != "backward_framewise":
            return 0
        return compute_idle_frames(
            action,
            self._action_spec(),
            eps_t=5e-3 / self._fps,
            eps_r=np.deg2rad(1.5) / self._fps,
            eps_g=1e-2,
            joint_threshold=5e-3 / self._fps,
            min_streak=3,
        )

    def _choose_mode(self) -> str:
        if self._mode == "joint":
            return random.choice(_MODE_CHOICES)
        return self._mode

    def _video_path(self, episode: dict[str, Any], video_key: str) -> Path:
        chunk_idx = int(
            episode.get(
                f"videos/{video_key}/chunk_index",
                episode.get(f"videos/{video_key}/episode_chunk", episode.get("data/chunk_index", 0)),
            )
        )
        file_idx = int(
            episode.get(
                f"videos/{video_key}/file_index",
                episode.get(f"videos/{video_key}/episode_file", episode.get("data/file_index", 0)),
            )
        )
        rel = self._info["video_path"].format(
            video_key=video_key,
            chunk_index=chunk_idx,
            file_index=file_idx,
            episode_chunk=chunk_idx,
            episode_file=file_idx,
        )
        return self._root / rel

    def _load_norm_stats(self) -> dict[str, torch.Tensor]:
        if self._norm_stats is None:
            self._norm_stats = {
                key: torch.from_numpy(value).float() for key, value in load_action_stats(str(self.stats_path())).items()
            }
        return self._norm_stats

    def _build_result(
        self,
        *,
        mode: str,
        video: torch.Tensor,
        action: torch.Tensor,
        ai_caption: str,
        has_initial_state: bool = False,
        **extras: Any,
    ) -> dict[str, Any]:
        # Idle detection reads the delta rows; the a0 row is an absolute pose, not a motion.
        idle_frames = self._compute_idle_frames(action[1:] if has_initial_state else action)
        # action_normalization=None -> use raw actions (no normalization), e.g. joint_pos.
        normalized_action = self.normalize(action, has_initial_state=has_initial_state)
        formatted_video = (video * 255.0).clamp(0.0, 255.0).to(torch.uint8).permute(1, 0, 2, 3)
        return {
            "ai_caption": ai_caption,
            "video": formatted_video,
            "action": normalized_action,
            "conditioning_fps": torch.tensor(self._fps, dtype=torch.long),
            "mode": mode,
            "domain_id": torch.tensor(self._domain_id, dtype=torch.long),
            "viewpoint": self._viewpoint,
            "idle_frames": torch.tensor(idle_frames, dtype=torch.long),
            # Only a0-enabled datasets carry the flag (absent == False for every consumer); see INITIAL_STATE_MODES.
            **({"has_initial_state": has_initial_state} if self._initial_state is not None else {}),
            **extras,
        }

    @property
    def _rows(self) -> list[dict[str, Any]]:
        """Flat, index-sorted list of every frame dict, built lazily on first access.

        Only datasets that don't build their own compact index (bridge / agibot /
        robomind) touch this; for them it materializes once and caches. Datasets with
        a bespoke index (e.g. DROIDLeRobotDataset) never read it, so they skip the
        ~13 min / tens-of-GB construction entirely.
        """
        if self._rows_cache is None:
            self._rows_cache = sorted(
                (
                    row
                    for path in sorted((self._root / "data").glob("chunk-*/file-*.parquet"))
                    for row in pq.read_table(path).to_pylist()
                ),
                key=lambda row: int(row["index"]),
            )
        return self._rows_cache

    def __len__(self) -> int:
        return max(0, (len(self._rows) - self._chunk_length + self._sample_stride - 1) // self._sample_stride)
