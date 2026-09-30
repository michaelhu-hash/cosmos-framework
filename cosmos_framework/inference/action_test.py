# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Action inference batch construction (CPU-only pieces)."""

from __future__ import annotations

import pytest

from cosmos_framework.inference.action import resolve_raw_action_dim
from cosmos_framework.inference.args import ModelMode


@pytest.mark.L0
def test_per_dataset_width_domains_take_the_spec_raw_action_dim() -> None:
    """hand_pose (Mecka / Cosmos3-Nano-HumanAction, 57D) has no canonical width: ID / wam need the spec's value."""
    assert resolve_raw_action_dim("hand_pose", ModelMode.INVERSE_DYNAMICS, 57) == 57
    assert resolve_raw_action_dim("hand_pose", ModelMode.WAM, 57) == 57
    assert resolve_raw_action_dim("hand_pose", ModelMode.FORWARD_DYNAMICS, None) is None  # width from the action file
    with pytest.raises(ValueError, match="pass raw_action_dim"):
        resolve_raw_action_dim("hand_pose", ModelMode.INVERSE_DYNAMICS, None)
    assert resolve_raw_action_dim("webhumanaction_hand", ModelMode.INVERSE_DYNAMICS, None) == 48
    assert resolve_raw_action_dim("webhumanaction_body", ModelMode.WAM, 57) == 57
    with pytest.raises(ValueError, match="contradicts"):
        resolve_raw_action_dim("webhumanaction_hand", ModelMode.INVERSE_DYNAMICS, 57)
