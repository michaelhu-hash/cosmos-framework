# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Statistics tool: output format loads through the readers' normalizer and round-trips."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from cosmos_framework.data.generator.action.utils.action_processing import load_action_normalizer
from cosmos_framework.scripts import compute_action_normalizer_stats as tool


@pytest.mark.L0
def test_stats_have_the_reader_format_and_quantile_semantics() -> None:
    rng = np.random.default_rng(0)
    rows = rng.normal(size=(5000, 4)) * np.array([1.0, 2.0, 0.5, 10.0]) + np.array([0.0, 1.0, -1.0, 5.0])
    stats = tool._stats(rows, 0.01, 0.99)
    assert set(stats) == {"mean", "std", "min", "max", "q01", "q99"} and all(len(v) == 4 for v in stats.values())
    np.testing.assert_allclose(stats["q01"], np.quantile(rows, 0.01, axis=0))
    np.testing.assert_allclose(stats["q99"], np.quantile(rows, 0.99, axis=0))
    with pytest.raises(ValueError):
        tool._stats(rows[:1], 0.01, 0.99)


@pytest.mark.L1
@pytest.mark.skipif(
    "COSMOS_HUMAN_HAND_POSE_LEROBOT_ROOT" not in os.environ,
    reason="set COSMOS_HUMAN_HAND_POSE_LEROBOT_ROOT to a Mecka hand-pose LeRobot export to run",
)
def test_tool_writes_loadable_delta_and_initial_state_stats(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = os.environ["COSMOS_HUMAN_HAND_POSE_LEROBOT_ROOT"]
    out, a0_out = tmp_path / "stats.json", tmp_path / "a0_stats.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--dataset",
            "human_hand_pose",
            "--root",
            root,
            "--output",
            str(out),
            "--initial-state-output",
            str(a0_out),
            "--snap-to-subtask",
            "--split",
            "full",
            "--max-windows",
            "50",
        ],
    )
    tool.main()
    for path in (out, a0_out):
        data = json.loads(path.read_text())
        assert set(data) == {"metadata", "global", "global_raw"} and len(data["global"]["q01"]) == 57
        normalizer = load_action_normalizer(
            "piecewise_asinh_rot", stats_path=path, apply_forward_clamp=False, expected_dim=57
        )
        sample = torch.tensor(data["global"]["mean"], dtype=torch.float32)[None] + 0.3 * torch.tensor(
            data["global"]["std"]
        )
        torch.testing.assert_close(
            normalizer.denormalize_action(normalizer.normalize_action(sample)), sample, atol=1e-5, rtol=1e-5
        )
    meta = json.loads(a0_out.read_text())["metadata"]
    assert meta["row"] == "initial_state" and meta["snap_to_subtask"] is True and meta["num_windows_used"] <= 50
