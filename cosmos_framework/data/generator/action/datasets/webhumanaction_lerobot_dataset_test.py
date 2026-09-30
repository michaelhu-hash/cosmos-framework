# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Unit tests for the Action100M (WebHumanAction) hand/body action builders and their normalizers."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from cosmos_framework.data.generator.action.datasets import human_hand_pose_lerobot_dataset as hand_pose_module
from cosmos_framework.data.generator.action.datasets import webhumanaction_lerobot_dataset as m
from cosmos_framework.data.generator.action.utils.action_processing import load_action_normalizer
from cosmos_framework.data.generator.action.utils.domain_utils import get_action_dim, get_domain_id
from cosmos_framework.data.generator.action.utils.human_pose_layout import (
    decode_human_pose_chains,
    decode_initial_state_row,
    split_initial_state,
)
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
        (m.HAND_INITIAL_STATE_NORMALIZER_PATH, 48),
        (m.BODY_INITIAL_STATE_NORMALIZER_PATH, 57),
        (hand_pose_module.HUMANACTION_INITIAL_STATE_NORMALIZER_PATH, 57),
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
    assert item["action_caption_attributes"]["dataset_name"] == "web_human_action_hand"
    assert torch.isfinite(item["action"]).all()
    raw = dataset.denormalize(item["action"])
    assert torch.isfinite(raw).all()
    _check_initial_state_end_to_end(
        m.WebHumanActionHandLeRobotDataset, os.environ["COSMOS_WEBHUMANACTION_HAND_LEROBOT_ROOT"], 48
    )


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
    assert item["action_caption_attributes"]["dataset_name"] == "web_human_action_body"
    assert torch.isfinite(item["action"]).all()
    _check_initial_state_end_to_end(
        m.WebHumanActionBodyLeRobotDataset, os.environ["COSMOS_WEBHUMANACTION_BODY_LEROBOT_ROOT"], 57
    )


def _check_initial_state_end_to_end(cls: type, root: str, dim: int) -> None:
    """Image2Action reader contract on real data: T+1 rows in ID/wam, flag set, row-aware (de)normalization."""
    dataset = cls(root, mode="inverse_dynamics", initial_state="predict")
    item = dataset[0]
    assert item["action"].shape == (73, dim) and item["has_initial_state"] is True
    assert torch.isfinite(item["action"]).all()
    raw = dataset.denormalize(item["action"])  # row count -> a0 routing
    assert raw.shape == (73, dim) and torch.isfinite(raw).all()
    torch.testing.assert_close(dataset.normalize(raw), item["action"], atol=1e-4, rtol=1e-4)
    # The same window WITHOUT a0 must equal rows 1.. of the a0 window (a0 is prepended, deltas unchanged).
    plain_dataset = cls(root, mode="inverse_dynamics")
    torch.testing.assert_close(raw[1:], plain_dataset.denormalize(plain_dataset[0]["action"]), atol=1e-5, rtol=1e-5)
    # Forward dynamics never carries a0.
    fd = cls(root, mode="forward_dynamics", initial_state="predict")[0]
    assert fd["action"].shape == (72, dim) and fd["has_initial_state"] is False


@pytest.mark.L0
def test_initial_state_row_is_the_absolute_frame0_pose() -> None:
    """a0 = [(head,) right wrist, right tips, left wrist, left tips] at frame 0, absolute, camera frame."""
    num_frames = 17
    t = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)[:, None]
    right = np.array([0.1, -0.2, 0.5], dtype=np.float32) + t * np.array([0.3, 0.0, 0.1], dtype=np.float32)
    left = np.array([-0.2, -0.1, 0.6], dtype=np.float32) + t * np.array([0.0, 0.2, 0.0], dtype=np.float32)
    head = np.array([0.0, -0.3, 0.1], dtype=np.float32) + t * np.array([0.05, 0.0, 0.0], dtype=np.float32)
    sample = _sample(num_frames, right_wrist=right, left_wrist=left, head_xyz=head)
    for build, layout, dim, has_head in (
        (m.build_webhumanaction_hand_action, m.HAND_LAYOUT, 48, False),
        (m.build_webhumanaction_body_action, m.BODY_LAYOUT, 57, True),
    ):
        plain = build(sample)
        window = build(sample, include_initial_state=True)
        assert window.shape == (num_frames, dim) and plain.shape == (num_frames - 1, dim)
        a0, rows = split_initial_state(window)
        np.testing.assert_array_equal(rows, plain)  # a0 is prepended; the delta rows are untouched
        anchors = decode_initial_state_row(a0, layout)
        # Aligned camera-frame wrist poses at frame 0 (identity joint rotations -> the alignment itself).
        np.testing.assert_allclose(anchors.wrist_poses[0][:3, 3], right[0], atol=1e-6)
        np.testing.assert_allclose(
            anchors.wrist_poses[0][:3, :3], m.WRIST_FRAME_ALIGN_ACTION100M_RIGHT[:3, :3], atol=1e-5
        )
        np.testing.assert_allclose(anchors.wrist_poses[1][:3, 3], left[0], atol=1e-6)
        np.testing.assert_allclose(
            anchors.wrist_poses[1][:3, :3], m.WRIST_FRAME_ALIGN_ACTION100M_LEFT[:3, :3], atol=1e-5
        )
        # Frame-0 fingertips in the wrist-0 frame: same rigid hand as the delta rows carry at frame 1.
        np.testing.assert_allclose(anchors.fingers_local[0].reshape(-1), rows[0, layout.fingers_slice(0)], atol=1e-5)
        if has_head:
            np.testing.assert_allclose(anchors.head_pose[:3, 3], head[0], atol=1e-6)
            np.testing.assert_allclose(anchors.head_pose[:3, :3], np.eye(3), atol=1e-6)
        else:
            assert anchors.head_pose is None
        # a0 as the only anchor + the deltas reproduce the camera-frame wrist trajectory (static camera).
        chains = decode_human_pose_chains(rows, layout, anchors, pose_convention=m.DEFAULT_POSE_CONVENTION)
        np.testing.assert_allclose(chains.wrist_poses[0][:, :3, 3], right, atol=1e-4)
        np.testing.assert_allclose(chains.wrist_poses[1][:, :3, 3], left, atol=1e-4)
        if has_head:
            np.testing.assert_allclose(chains.head_poses[:, :3, 3], head, atol=1e-4)


@pytest.mark.L0
def test_initial_state_stats_match_the_arm_widths() -> None:
    for path, dim in (
        (m.HAND_INITIAL_STATE_NORMALIZER_PATH, 48),
        (m.BODY_INITIAL_STATE_NORMALIZER_PATH, 57),
        (hand_pose_module.HUMANACTION_INITIAL_STATE_NORMALIZER_PATH, 57),
    ):
        stats = json.loads(path.read_text())["global"]
        assert all(len(stats[key]) == dim for key in ("mean", "std", "q01", "q99"))


# Training prompt of the released Cosmos3-Nano-HumanAction checkpoints (internal recipe: plain caption followed by the
# dataset's caption-semantics sentences and the resolution; no JSON wrapping, duration / fps or idle-frame text).
_RELEASED_HAND_PROMPT_TAIL = (
    "The video is captured from a real-world environment. The video shows a human actor. "
    "This video is captured from a static perspective looking towards the actor. "
    "The action performed by the human actor is defined as the wrist and fingertip motion of the actor's hands, "
    "mapped to the right and left wrist-pose and fingertip components. This video is of 480x832 resolution."
)


@pytest.mark.L0
def test_readers_register_the_released_caption_semantics() -> None:
    from cosmos_framework.data.generator.action.action_caption_attribute_adapter import ACTION_CAPTION_ATTRIBUTE_ADAPTER

    assert m.WebHumanActionHandLeRobotDataset._ACTION_CAPTION_DATASET_NAME == "web_human_action_hand"
    assert m.WebHumanActionBodyLeRobotDataset._ACTION_CAPTION_DATASET_NAME == "web_human_action_body"
    for name in ("web_human_action_hand", "web_human_action_body", "embodiment_a"):
        assert ACTION_CAPTION_ATTRIBUTE_ADAPTER.supports(name)
    attrs = ACTION_CAPTION_ATTRIBUTE_ADAPTER.resolve("web_human_action_hand", fps=15.0, observation_count=73, view_count=1)
    assert attrs["view_postfix"] == "This video is captured from a static perspective looking towards the actor."
    assert attrs["action_transition_count"] == 72


@pytest.mark.L0
def test_humanaction_factories_default_to_the_released_prompt() -> None:
    import inspect

    from cosmos_framework.data.generator.action.datasets import action_sft_dataset as sft_module

    for fn in (
        sft_module.get_action_webhumanaction_hand_sft_dataset,
        sft_module.get_action_webhumanaction_body_sft_dataset,
        sft_module.get_action_human_hand_pose_sft_dataset,
    ):
        defaults = {k: v.default for k, v in inspect.signature(fn).parameters.items()}
        assert defaults["append_action_caption_semantics"] is True, fn.__name__
        assert defaults["format_prompt_as_json"] is False, fn.__name__
        assert defaults["append_duration_fps_timestamps"] is False and defaults["append_idle_frames"] is False, fn.__name__
        assert defaults["append_resolution_info"] is True, fn.__name__
    mecka = inspect.signature(sft_module.get_action_human_hand_pose_sft_dataset).parameters
    assert mecka["action_caption_dataset_name"].default == "embodiment_a"


@pytest.mark.L0
def test_humanaction_sft_prompt_matches_the_released_recipe() -> None:
    """The factory transform turns a reader item into the exact training prompt of the released checkpoints."""
    from cosmos_framework.data.generator.action.action_caption_attribute_adapter import ACTION_CAPTION_ATTRIBUTE_ADAPTER
    from cosmos_framework.data.generator.action.datasets.action_sft_dataset import _humanaction_sft

    class _OneItem(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 1

        def __getitem__(self, idx: int) -> dict:
            return {
                "ai_caption": "The person places a ruler on the fabric.",
                "video": torch.zeros(3, 73, 480, 832, dtype=torch.uint8),  # [C,T,H,W], reader layout
                "action": torch.zeros(72, 48),
                "conditioning_fps": torch.tensor(15),
                "mode": "forward_dynamics",
                "domain_id": torch.tensor(31),
                "viewpoint": "ego_view",
                "idle_frames": torch.tensor(0),
                "action_caption_attributes": ACTION_CAPTION_ATTRIBUTE_ADAPTER.resolve(
                    "web_human_action_hand", fps=15.0, observation_count=73, view_count=1
                ),
            }

    sft = _humanaction_sft(
        _OneItem(),
        resolution="480",
        max_action_dim=64,
        tokenizer_config=None,
        cfg_dropout_rate=0.0,
        append_viewpoint_info=True,
        append_action_caption_semantics=True,
        append_duration_fps_timestamps=False,
        append_resolution_info=True,
        append_idle_frames=False,
        format_prompt_as_json=False,
        iterable_shuffle=False,
        episode_shuffle_seed=0,
    )
    item = sft[0]
    assert item["ai_caption"] == "The person places a ruler on the fabric. " + _RELEASED_HAND_PROMPT_TAIL
    assert item["action"].shape == (72, 64)
    assert "action_caption_attributes" not in item
