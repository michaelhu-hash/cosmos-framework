# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Compute action normalizer statistics for the Cosmos3-Nano-HumanAction LeRobot readers.

Writes the ``normalizer_stats/*.json`` format the readers consume (``metadata`` / ``global`` / ``global_raw`` with
per-dimension ``mean, std, min, max, q01, q99``) from the RAW (un-normalized) action windows of your own data, so a
post-training run on new data can use matching ``piecewise_asinh_rot`` statistics. With ``--initial-state-output``
the frame-0 initial-state rows (Image2Action a0) get their own statistics file, as the a0 recipes require.

Example::

    python -m cosmos_framework.scripts.compute_action_normalizer_stats \\
        --dataset webhumanaction_hand --root /data/my_hands_lerobot \\
        --output my_hand_stats.json --initial-state-output my_hand_initial_state_stats.json

Only the action columns are read (no video decoding), so this runs at thousands of windows per second.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cosmos_framework.data.generator.action.datasets.human_hand_pose_lerobot_dataset import HumanHandPoseLeRobotDataset
from cosmos_framework.data.generator.action.datasets.webhumanaction_lerobot_dataset import (
    WebHumanActionBodyLeRobotDataset,
    WebHumanActionHandLeRobotDataset,
)

DATASETS = {
    "webhumanaction_hand": WebHumanActionHandLeRobotDataset,
    "webhumanaction_body": WebHumanActionBodyLeRobotDataset,
    "human_hand_pose": HumanHandPoseLeRobotDataset,
}
STAT_KEYS = ("mean", "std", "min", "max")


def _stats(rows: np.ndarray, quantile_low: float, quantile_high: float) -> dict[str, list[float]]:
    """Per-dimension statistics of ``rows`` (``[N, D]``); the two quantile keys are named ``q01`` / ``q99`` by the
    readers' convention regardless of the actual quantile levels (recorded in ``metadata``)."""
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] < 2:
        raise ValueError(f"need at least 2 rows to compute statistics, got shape {rows.shape}")
    return {
        "mean": rows.mean(0).tolist(),
        "std": rows.std(0).tolist(),
        "min": rows.min(0).tolist(),
        "max": rows.max(0).tolist(),
        "q01": np.quantile(rows, quantile_low, axis=0).tolist(),
        "q99": np.quantile(rows, quantile_high, axis=0).tolist(),
    }


def collect_raw_rows(
    dataset: Any,
    *,
    max_windows: int | None,
    max_rows: int,
    include_initial_state: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    """Raw delta rows ``[N, D]`` (and a0 rows ``[W, D]`` when requested) over up to ``max_windows`` evenly spaced
    windows; the delta rows are reservoir-capped at ``max_rows`` to bound memory."""
    num_windows = len(dataset)
    if num_windows == 0:
        raise ValueError("the dataset has no windows (too few frames per episode for the chunk length?)")
    indices = np.arange(num_windows)
    if max_windows is not None and max_windows < num_windows:
        indices = np.linspace(0, num_windows - 1, max_windows).round().astype(int)
    rng = np.random.default_rng(seed)
    reservoir: list[np.ndarray] = []
    a0_rows: list[np.ndarray] = []
    seen_rows = 0
    kept: np.ndarray | None = None
    for idx in indices:
        rows = dataset._window_rows(int(idx))
        action = dataset._build_raw_action(rows, include_initial_state=include_initial_state)
        action = action.numpy() if isinstance(action, torch.Tensor) else np.asarray(action)
        if include_initial_state:
            a0_rows.append(action[0])
            action = action[1:]
        if kept is None:
            kept = np.empty((0, action.shape[1]), dtype=np.float32)
        # Reservoir over rows: keep the first max_rows, then replace uniformly at random.
        for row in action:
            if kept.shape[0] < max_rows:
                reservoir.append(row.astype(np.float32))
                if len(reservoir) == max_rows:
                    kept = np.stack(reservoir)
            else:
                j = int(rng.integers(0, seen_rows + 1))
                if j < max_rows:
                    kept[j] = row
            seen_rows += 1
    if kept is None or kept.shape[0] == 0:
        kept = np.stack(reservoir)
    return kept, (np.stack(a0_rows) if a0_rows else None), seen_rows


def compute(args: argparse.Namespace) -> None:
    cls = DATASETS[args.dataset]
    kwargs: dict[str, Any] = dict(
        fps=args.fps,
        chunk_length=args.chunk_length,
        mode="inverse_dynamics",  # a0 is only built for the action-predicting modes
        pose_convention=args.pose_convention,
        action_normalization=None,  # raw actions
        split=args.split,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        snap_to_subtask=args.snap_to_subtask,
        sample_stride=args.sample_stride,
    )
    if args.initial_state_output is not None:
        kwargs["initial_state"] = "predict"
    dataset = cls(args.root, **kwargs)
    t0 = time.time()
    delta_rows, a0_rows, seen_rows = collect_raw_rows(
        dataset,
        max_windows=args.max_windows,
        max_rows=args.max_rows,
        include_initial_state=args.initial_state_output is not None,
        seed=args.seed,
    )
    metadata = {
        "embodiment_type": dataset.domain_name,
        "dataset_class": cls.__name__,
        "dataset_root": str(args.root),
        "pose_convention": args.pose_convention,
        "rotation_format": "rot6d",
        "action_dim": int(dataset.action_dim),
        "skip_rotation_dims": [],
        "chunk_length": args.chunk_length,
        "dataset_fps": [args.fps],
        "split": args.split,
        "val_ratio": args.val_ratio,
        "snap_to_subtask": bool(args.snap_to_subtask),
        "num_windows_total": len(dataset),
        "num_windows_used": int(min(len(dataset), args.max_windows) if args.max_windows else len(dataset)),
        "num_rows_seen": int(seen_rows),
        "num_rows_stats": int(delta_rows.shape[0]),
        "quantile_low": args.quantile_low,
        "quantile_high": args.quantile_high,
        "quantile_fields": {"q01": args.quantile_low, "q99": args.quantile_high},
        "seconds": round(time.time() - t0, 1),
    }
    stats = _stats(delta_rows, args.quantile_low, args.quantile_high)
    Path(args.output).write_text(json.dumps({"metadata": metadata, "global": stats, "global_raw": stats}, indent=1))
    print(
        f"wrote {args.output}: {delta_rows.shape[0]} delta rows x {delta_rows.shape[1]} dims from {metadata['num_windows_used']} windows"
    )
    if args.initial_state_output is not None:
        assert a0_rows is not None
        a0_stats = _stats(a0_rows, args.quantile_low, args.quantile_high)
        a0_meta = {**metadata, "row": "initial_state", "num_rows_stats": int(a0_rows.shape[0])}
        Path(args.initial_state_output).write_text(
            json.dumps({"metadata": a0_meta, "global": a0_stats, "global_raw": a0_stats}, indent=1)
        )
        print(f"wrote {args.initial_state_output}: {a0_rows.shape[0]} initial-state rows x {a0_rows.shape[1]} dims")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--root", required=True, help="LeRobot v3 dataset root (meta/ + data/ + videos/)")
    parser.add_argument("--output", required=True, help="output JSON for the delta-row (rows 1..T) statistics")
    parser.add_argument("--initial-state-output", default=None, help="output JSON for the frame-0 a0 row statistics")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--chunk-length", type=int, default=72)
    parser.add_argument("--pose-convention", default="backward_chunk_anchored_16f")
    parser.add_argument("--split", default="train", choices=("full", "train", "val"))
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--snap-to-subtask", action="store_true", help="one window per subtask (the Mecka recipe)")
    parser.add_argument("--sample-stride", type=int, default=1, help="dense-window start stride (source rows)")
    parser.add_argument("--max-windows", type=int, default=100_000, help="evenly spaced windows to read (None = all)")
    parser.add_argument("--max-rows", type=int, default=2_000_000, help="reservoir cap on delta rows kept in memory")
    parser.add_argument("--quantile-low", type=float, default=0.01)
    parser.add_argument("--quantile-high", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    compute(build_parser().parse_args())


if __name__ == "__main__":
    main()
