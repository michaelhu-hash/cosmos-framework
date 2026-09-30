# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Action inference batch construction (CPU-only pieces)."""

from __future__ import annotations

import pytest
import torch

from cosmos_framework.data.generator.action.utils.transforms import build_sequence_plan_from_mode
from cosmos_framework.inference.action import _load_actions
from cosmos_framework.inference.args import ActionDataArgs, ActionDataOverrides, ModelMode


@pytest.mark.L0
@pytest.mark.parametrize("mode", [ModelMode.INVERSE_DYNAMICS, ModelMode.WAM])
def test_predict_initial_state_adds_the_a0_row_to_the_generated_actions(mode: ModelMode) -> None:
    plain, dim = _load_actions(None, mode, action_chunk_size=72, max_action_dim=64, raw_action_dim=48)
    a0, dim_a0 = _load_actions(
        None, mode, action_chunk_size=72, max_action_dim=64, raw_action_dim=48, predict_initial_state=True
    )
    assert plain.shape == (72, 64) and a0.shape == (73, 64) and dim == dim_a0 == 48
    assert torch.count_nonzero(a0) == 0
    # The plan the batch builder derives: a0 + one delta per frame, nothing conditioning, a0 on frame 0.
    plan = build_sequence_plan_from_mode(mode.value, video_length=73, action_length=73, predict_initial_state=True)
    assert (
        plan.predict_initial_state and plan.condition_frame_indexes_action == [] and plan.action_start_frame_offset == 0
    )
    legacy = build_sequence_plan_from_mode(mode.value, video_length=73, action_length=72)
    assert not legacy.predict_initial_state and legacy.action_start_frame_offset == 1


@pytest.mark.L0
def test_predict_initial_state_is_rejected_for_forward_dynamics(tmp_path) -> None:
    action_file = tmp_path / "actions.json"
    action_file.write_text("[[0.0, 0.0]]")
    with pytest.raises(ValueError, match="inverse_dynamics / wam"):
        _load_actions(action_file, ModelMode.FORWARD_DYNAMICS, 1, 64, None, predict_initial_state=True)


@pytest.mark.L0
def test_predict_initial_state_arg_defaults_off() -> None:
    assert ActionDataArgs().predict_initial_state is False
    assert ActionDataOverrides().predict_initial_state is None
    assert ActionDataOverrides(predict_initial_state=True).predict_initial_state is True
