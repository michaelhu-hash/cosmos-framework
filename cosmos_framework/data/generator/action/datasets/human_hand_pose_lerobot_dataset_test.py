# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Mecka / hand_pose reader: Image2Action (a0) contract on a real LeRobot export."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from cosmos_framework.data.generator.action.datasets import human_hand_pose_lerobot_dataset as m
from cosmos_framework.data.generator.action.utils.human_pose_layout import (
    decode_human_pose_chains,
    decode_initial_state_row,
    split_initial_state,
)


@pytest.mark.L0
def test_hand_pose_layout_matches_the_reader_width() -> None:
    assert m.HAND_POSE_LAYOUT.action_dim == 57 and m.HAND_POSE_LAYOUT.has_camera
    assert m.HUMANACTION_INITIAL_STATE_NORMALIZER_PATH.exists()


@pytest.mark.L0
def test_initial_state_requires_an_asinh_family_normalizer(tmp_path) -> None:
    with pytest.raises(NotImplementedError):
        m.HumanHandPoseLeRobotDataset(str(tmp_path), action_normalization="quantile", initial_state="predict")


@pytest.mark.L0
def test_caption_semantics_protocol_is_validated_before_any_io(tmp_path) -> None:
    with pytest.raises(ValueError, match="protocol"):
        m.HumanHandPoseLeRobotDataset(str(tmp_path), action_caption_dataset_name="not_a_protocol")


@pytest.mark.L1
@pytest.mark.skipif(
    "COSMOS_HUMAN_HAND_POSE_LEROBOT_ROOT" not in os.environ,
    reason="set COSMOS_HUMAN_HAND_POSE_LEROBOT_ROOT to a Mecka hand-pose LeRobot export to run",
)
def test_initial_state_end_to_end() -> None:
    root = os.environ["COSMOS_HUMAN_HAND_POSE_LEROBOT_ROOT"]
    kwargs = dict(
        chunk_length=72,
        pose_convention="backward_chunk_anchored_16f",
        action_normalization="piecewise_asinh_rot",
        stats_path=m.HUMANACTION_NORMALIZER_PATH,
    )
    dataset = m.HumanHandPoseLeRobotDataset(root, mode="inverse_dynamics", initial_state="predict", **kwargs)
    item = dataset[0]
    assert item["action"].shape == (73, 57) and item["has_initial_state"] is True
    assert "action_caption_attributes" not in item  # base-Nano reader default: no caption semantics
    semantic = m.HumanHandPoseLeRobotDataset(
        root, mode="inverse_dynamics", initial_state="predict", action_caption_dataset_name="embodiment_a", **kwargs
    )[0]
    attrs = semantic["action_caption_attributes"]
    assert attrs["dataset_name"] == "embodiment_a" and attrs["observation_count"] == 73
    raw = dataset.denormalize(item["action"])
    torch.testing.assert_close(dataset.normalize(raw), item["action"], atol=1e-4, rtol=1e-4)
    a0, rows = split_initial_state(raw.numpy())
    anchors = decode_initial_state_row(a0, m.HAND_POSE_LAYOUT)
    np.testing.assert_allclose(anchors.camera_pose, np.eye(4), atol=1e-5)  # a0 carries no ego-motion
    chains = decode_human_pose_chains(rows, m.HAND_POSE_LAYOUT, anchors, pose_convention="backward_chunk_anchored_16f")
    assert chains.num_frames == 73 and np.isfinite(chains.wrist_poses[0]).all()
    plain = m.HumanHandPoseLeRobotDataset(root, mode="inverse_dynamics", **kwargs)
    torch.testing.assert_close(raw[1:], plain.denormalize(plain[0]["action"]), atol=1e-5, rtol=1e-5)
    fd = m.HumanHandPoseLeRobotDataset(root, mode="forward_dynamics", initial_state="predict", **kwargs)[0]
    assert fd["action"].shape == (72, 57) and fd["has_initial_state"] is False
