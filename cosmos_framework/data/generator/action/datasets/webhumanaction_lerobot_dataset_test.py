# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Unit tests for the Action100M (WebHumanAction) hand/body action builders and their normalizers."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from cosmos_framework.data.generator.action.datasets import human_hand_pose_lerobot_dataset as hand_pose_module
from cosmos_framework.data.generator.action.datasets import webhumanaction_lerobot_dataset as m
from cosmos_framework.data.generator.action.utils.action_processing import load_action_normalizer
from cosmos_framework.data.generator.action.utils.domain_utils import get_action_dim, get_domain_id
from cosmos_framework.data.generator.action.utils.pose_utils import build_abs_pose_from_components, pose_abs_to_rel

_IDENTITY_ROT6D = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
_IDENTITY_QUAT_XYZW = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _hand_state(num_frames: int, wrist_xyz: np.ndarray, spread: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """21-joint hand with identity joint rotations; joint ``j`` sits ``spread * j`` along +x from the wrist."""
    joints = np.zeros((num_frames, m.NUM_JOINTS, 3), dtype=np.float32)
    joints[:, :, :] = wrist_xyz[:, None, :]
    joints[:, :, 0] += spread * np.arange(m.NUM_JOINTS, dtype=np.float32)[None, :]
    rotations = np.tile(_IDENTITY_QUAT_XYZW, (num_frames, m.NUM_JOINTS, 1)).astype(np.float32)
    return joints.reshape(num_frames, -1), rotations.reshape(num_frames, -1)


def _sample(
    num_frames: int,
    *,
    right_wrist: np.ndarray,
    left_wrist: np.ndarray,
    camera_xyz: np.ndarray | None = None,
    head_xyz: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    right_pos, right_rot = _hand_state(num_frames, right_wrist)
    left_pos, left_rot = _hand_state(num_frames, left_wrist)
    camera_xyz = np.zeros((num_frames, 3), dtype=np.float32) if camera_xyz is None else camera_xyz
    head_xyz = np.zeros((num_frames, 3), dtype=np.float32) if head_xyz is None else head_xyz
    return {
        m.CAMERA_POSITION_KEY: camera_xyz,
        m.CAMERA_ROTATION_KEY: np.tile(_IDENTITY_QUAT_XYZW, (num_frames, 1)),
        m.HEAD_POSITION_KEY: head_xyz,
        m.HEAD_ROTATION_KEY: np.tile(_IDENTITY_QUAT_XYZW, (num_frames, 1)),
        m.HAND_RIGHT_POSITION_KEY: right_pos,
        m.HAND_RIGHT_ROTATION_KEY: right_rot,
        m.HAND_LEFT_POSITION_KEY: left_pos,
        m.HAND_LEFT_ROTATION_KEY: left_rot,
    }


@pytest.mark.L0
def test_domain_tables_register_both_arms() -> None:
    assert get_domain_id("webhumanaction_hand") == 31
    assert get_domain_id("webhumanaction_body") == 24
    assert get_action_dim("webhumanaction_hand") == m.HAND_ACTION_DIM == 48
    assert get_action_dim("webhumanaction_body") == m.BODY_ACTION_DIM == 57


@pytest.mark.L0
def test_static_scene_gives_identity_deltas_and_constant_fingertips() -> None:
    num_frames = 17
    still = np.tile(np.array([0.1, -0.2, 0.5], dtype=np.float32), (num_frames, 1))
    sample = _sample(num_frames, right_wrist=still, left_wrist=still + 0.3)
    hand = m.build_webhumanaction_hand_action(sample, pose_convention="backward_framewise")
    body = m.build_webhumanaction_body_action(sample, pose_convention="backward_framewise")
    assert hand.shape == (num_frames - 1, 48) and body.shape == (num_frames - 1, 57)
    identity_rows = np.broadcast_to(_IDENTITY_ROT6D, (num_frames - 1, 6))
    for block_start in (0, 24):  # right / left wrist pose blocks of the hand action
        np.testing.assert_allclose(hand[:, block_start : block_start + 3], 0.0, atol=1e-6)
        np.testing.assert_allclose(hand[:, block_start + 3 : block_start + 9], identity_rows, atol=1e-6)
    # Fingertips are fixed in the wrist frame, so every row repeats the same 15 values.
    right_tips, left_tips = hand[:, 9:24], hand[:, 33:48]
    np.testing.assert_allclose(right_tips, np.broadcast_to(right_tips[:1], right_tips.shape), atol=1e-6)
    # Fingertip j lies spread*j along camera +x. In the aligned wrist frames that camera axis is -z for the right
    # hand and +z for the left hand (see WRIST_FRAME_ALIGN_ACTION100M_{RIGHT,LEFT}).
    offsets = 0.02 * np.array(m.FINGERTIP_JOINT_IDXS, dtype=np.float32)
    expected_right = np.zeros((5, 3), dtype=np.float32)
    expected_right[:, 2] = -offsets
    expected_left = np.zeros((5, 3), dtype=np.float32)
    expected_left[:, 2] = offsets
    np.testing.assert_allclose(right_tips[0].reshape(5, 3), expected_right, atol=1e-5)
    np.testing.assert_allclose(left_tips[0].reshape(5, 3), expected_left, atol=1e-5)
    # The head block of the body action is the identity delta for a static head/camera.
    np.testing.assert_allclose(body[:, :3], 0.0, atol=1e-6)
    np.testing.assert_allclose(body[:, 3:9], identity_rows, atol=1e-6)
    # Body hand blocks equal the hand action (same chain, camera block replaced by the head block).
    np.testing.assert_allclose(body[:, 9:], hand, atol=1e-6)


@pytest.mark.L0
def test_head_block_is_the_world_frame_head_delta() -> None:
    num_frames = 17
    still = np.tile(np.array([0.1, 0.0, 0.5], dtype=np.float32), (num_frames, 1))
    camera = np.zeros((num_frames, 3), dtype=np.float32)
    camera[:, 2] = 0.01 * np.arange(num_frames)  # camera drifts along +z
    head = np.zeros((num_frames, 3), dtype=np.float32)
    head[:, 0] = 0.005 * np.arange(num_frames)  # head drifts along camera +x
    sample = _sample(num_frames, right_wrist=still, left_wrist=still, camera_xyz=camera, head_xyz=head)
    body = m.build_webhumanaction_body_action(sample, pose_convention="backward_framewise")
    quat = np.tile(_IDENTITY_QUAT_XYZW, (num_frames, 1))
    head_world = build_abs_pose_from_components(camera, quat, "quat_xyzw") @ build_abs_pose_from_components(
        head, quat, "quat_xyzw"
    )
    expected = pose_abs_to_rel(head_world, rotation_format="rot6d", pose_convention="backward_framewise")
    np.testing.assert_allclose(body[:, :9], expected, atol=1e-6)
    per_step = np.broadcast_to(np.array([0.005, 0.0, 0.01], dtype=np.float32), (num_frames - 1, 3))
    np.testing.assert_allclose(body[:, :3], per_step, atol=1e-6)


@pytest.mark.L0
def test_chunk_anchored_16f_resets_the_anchor_every_16_steps() -> None:
    num_frames = 73  # the HumanAction chunk: 72 transitions
    wrist = np.zeros((num_frames, 3), dtype=np.float32)
    wrist[:, 0] = 0.01 * np.arange(num_frames)  # constant velocity along +x
    sample = _sample(num_frames, right_wrist=wrist, left_wrist=wrist)
    hand = m.build_webhumanaction_hand_action(sample, pose_convention="backward_chunk_anchored_16f")
    assert hand.shape == (72, 48)
    # Rows are relative to the anchor frame of their 16-step block: 0.01, 0.02, ..., 0.16, then again 0.01, ...
    # The delta is expressed in the anchor's aligned wrist frame, where camera +x is -z (right) / +z (left).
    expected = np.tile(0.01 * np.arange(1, 17, dtype=np.float32), 5)[:72]
    np.testing.assert_allclose(hand[:, 2], -expected, atol=1e-6)
    np.testing.assert_allclose(hand[:, 24 + 2], expected, atol=1e-6)
    np.testing.assert_allclose(hand[:, [0, 1, 24, 25]], 0.0, atol=1e-6)
    # Against the shared helper directly.
    quat = np.tile(_IDENTITY_QUAT_XYZW, (num_frames, 1))
    wrist_world = build_abs_pose_from_components(wrist, quat, "quat_xyzw") @ m.WRIST_FRAME_ALIGN_ACTION100M_RIGHT
    np.testing.assert_allclose(
        hand[:, :9],
        pose_abs_to_rel(wrist_world, rotation_format="rot6d", pose_convention="backward_chunk_anchored_16f"),
        atol=1e-6,
    )


@pytest.mark.L0
@pytest.mark.parametrize(
    ("stats_path", "dim"),
    [
        (m.HAND_NORMALIZER_PATH, 48),
        (m.BODY_NORMALIZER_PATH, 57),
        (hand_pose_module.HUMANACTION_NORMALIZER_PATH, 57),
    ],
)
def test_humanaction_normalizers_load_and_round_trip(stats_path: Path, dim: int) -> None:
    assert stats_path.exists(), stats_path
    normalizer = load_action_normalizer(
        "piecewise_asinh_rot", stats_path=stats_path, apply_forward_clamp=False, expected_dim=dim
    )
    raw = torch.randn(72, dim) * 0.05
    normalized = normalizer.normalize_action(raw)
    assert normalized.shape == (72, dim)
    assert torch.isfinite(normalized).all()
    torch.testing.assert_close(normalizer.denormalize_action(normalized), raw, atol=1e-5, rtol=1e-5)


@pytest.mark.L0
def test_hand_pose_reader_exposes_the_humanaction_recipe_knobs() -> None:
    # The same LeRobot data serves base Cosmos3-Nano (framewise, quantile) and Cosmos3-Nano-HumanAction
    # (72-step chunk-anchored, piecewise asinh); the reader must accept both.
    import inspect

    params = inspect.signature(hand_pose_module.HumanHandPoseLeRobotDataset.__init__).parameters
    assert "stats_path" in params and "pose_convention" in params and "action_normalization" in params
    assert "backward_chunk_anchored_16f" in hand_pose_module.PoseConvention.__args__


@pytest.mark.L1
@pytest.mark.skipif(
    "COSMOS_WEBHUMANACTION_HAND_LEROBOT_ROOT" not in os.environ,
    reason="set COSMOS_WEBHUMANACTION_HAND_LEROBOT_ROOT to an Action100M hand LeRobot export to run",
)
def test_hand_dataset_end_to_end() -> None:
    dataset = m.WebHumanActionHandLeRobotDataset(os.environ["COSMOS_WEBHUMANACTION_HAND_LEROBOT_ROOT"])
    assert len(dataset) > 0
    item = dataset[0]
    assert item["action"].shape == (72, 48)
    assert item["video"].shape[1] == 73 and item["video"].dtype == torch.uint8
    assert int(item["domain_id"]) == 31
    assert torch.isfinite(item["action"]).all()
    raw = dataset.denormalize(item["action"])
    assert torch.isfinite(raw).all()


@pytest.mark.L1
@pytest.mark.skipif(
    "COSMOS_WEBHUMANACTION_BODY_LEROBOT_ROOT" not in os.environ,
    reason="set COSMOS_WEBHUMANACTION_BODY_LEROBOT_ROOT to an Action100M body LeRobot export to run",
)
def test_body_dataset_end_to_end() -> None:
    dataset = m.WebHumanActionBodyLeRobotDataset(os.environ["COSMOS_WEBHUMANACTION_BODY_LEROBOT_ROOT"])
    assert len(dataset) > 0
    item = dataset[0]
    assert item["action"].shape == (72, 57)
    assert int(item["domain_id"]) == 24
    assert torch.isfinite(item["action"]).all()
