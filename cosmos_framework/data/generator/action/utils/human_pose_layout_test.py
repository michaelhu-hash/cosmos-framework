# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from cosmos_framework.data.generator.action.utils.human_pose_layout import (
    FINGERTIP_JOINT_IDXS,
    HumanPoseAnchors,
    HumanPoseLayout,
    decode_human_pose_chains,
    decode_initial_state_row,
    encode_initial_state_row,
    split_initial_state,
)
from cosmos_framework.data.generator.action.utils.pose_utils import pose_abs_to_rel

_J = len(FINGERTIP_JOINT_IDXS)
_LAYOUTS = {
    "hand_pose": HumanPoseLayout.hand_pose(),
    "hand_pose_camera_free": HumanPoseLayout.hand_pose_camera_free(),
    "body_head_wrists": HumanPoseLayout.body_head_wrists(),
}


def _random_pose(rng: np.random.Generator, *, translation_scale: float = 0.5) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = R.random(random_state=int(rng.integers(1 << 31))).as_matrix()
    pose[:3, 3] = rng.normal(scale=translation_scale, size=3)
    return pose


def _random_walk(
    rng: np.random.Generator, start: np.ndarray, num_frames: int, *, step_m: float, step_rad: float
) -> np.ndarray:
    """``(num_frames, 4, 4)`` SE(3) random walk starting at ``start``."""
    poses = [start]
    for _ in range(num_frames - 1):
        delta = np.eye(4)
        delta[:3, :3] = R.from_rotvec(rng.normal(scale=step_rad, size=3)).as_matrix()
        delta[:3, 3] = rng.normal(scale=step_m, size=3)
        poses.append(poses[-1] @ delta)
    return np.stack(poses)


def _anchors(rng: np.random.Generator, layout: HumanPoseLayout) -> HumanPoseAnchors:
    return HumanPoseAnchors(
        wrist_poses=(_random_pose(rng), _random_pose(rng)),
        fingers_local=(rng.normal(scale=0.05, size=(_J, 3)), rng.normal(scale=0.05, size=(_J, 3))),
        head_pose=_random_pose(rng) if layout.has_head else None,
    )


def _assert_anchors_close(got: HumanPoseAnchors, want: HumanPoseAnchors, *, atol: float = 1e-5) -> None:
    for hand in range(2):
        np.testing.assert_allclose(got.wrist_poses[hand], want.wrist_poses[hand], atol=atol)
        np.testing.assert_allclose(got.fingers_local[hand], want.fingers_local[hand], atol=1e-6)
    if want.head_pose is None:
        assert got.head_pose is None
    else:
        np.testing.assert_allclose(got.head_pose, want.head_pose, atol=atol)


class _Scene:
    """Synthetic camera-frame ground truth + the rows a dataset would encode from it (a0 + T deltas)."""

    def __init__(
        self, layout: HumanPoseLayout, pose_convention: str, *, num_steps: int, moving_camera: bool, seed: int
    ):
        rng = np.random.default_rng(seed)
        n = num_steps + 1
        self.layout, self.pose_convention = layout, pose_convention
        # Camera-to-world; identity walk == static camera (frame-0 camera frame is the world frame).
        self.camera_c2w = (
            _random_walk(rng, _random_pose(rng, translation_scale=1.0), n, step_m=0.01, step_rad=0.01)
            if moving_camera
            else np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
        )
        # Camera-frame chains (what a0 + the decoder are defined in).
        self.wrists_cam = [
            _random_walk(rng, _random_pose(rng, translation_scale=0.3), n, step_m=0.01, step_rad=0.02) for _ in range(2)
        ]
        self.offsets = [rng.normal(scale=0.05, size=(_J, 3)) for _ in range(2)]  # rigid fingertips in the wrist frame
        self.tips_cam = [
            np.einsum("tij,nj->tni", w[:, :3, :3], off) + w[:, None, :3, 3]
            for w, off in zip(self.wrists_cam, self.offsets)
        ]
        self.head_cam = _random_walk(rng, _random_pose(rng, translation_scale=0.2), n, step_m=0.005, step_rad=0.01)
        # Dataset encoding: pose blocks are WORLD-frame deltas, fingertips per-frame wrist-frame positions (rows 1..T).
        rf = layout.rotation_format
        blocks = []
        if layout.has_camera:
            blocks.append(pose_abs_to_rel(self.camera_c2w, rotation_format=rf, pose_convention=pose_convention))
        if layout.has_head:
            blocks.append(
                pose_abs_to_rel(self.camera_c2w @ self.head_cam, rotation_format=rf, pose_convention=pose_convention)
            )
        for hand in range(2):
            blocks.append(
                pose_abs_to_rel(
                    self.camera_c2w @ self.wrists_cam[hand], rotation_format=rf, pose_convention=pose_convention
                )
            )
            local = np.einsum(
                "tji,tnj->tni",
                self.wrists_cam[hand][1:, :3, :3],
                self.tips_cam[hand][1:] - self.wrists_cam[hand][1:, None, :3, 3],
            )
            blocks.append(local.reshape(num_steps, -1))
        self.rows = np.concatenate(blocks, axis=-1).astype(np.float32)  # [T,D]
        self.anchors = HumanPoseAnchors(
            wrist_poses=(self.wrists_cam[0][0], self.wrists_cam[1][0]),
            fingers_local=(self.offsets[0], self.offsets[1]),
            head_pose=self.head_cam[0] if layout.has_head else None,
        )
        self.a0 = encode_initial_state_row(layout, self.anchors)  # [D]
        self.window = np.concatenate([self.a0[None], self.rows], axis=0)  # [T+1,D]

    def expected_in_camera0(self, chain_cam: np.ndarray) -> np.ndarray:
        """Camera-frame chain -> frame-0 camera frame (the decoder's output frame)."""
        return np.linalg.inv(self.camera_c2w[0])[None] @ self.camera_c2w @ chain_cam


@pytest.mark.L0
def test_layout_widths_and_slices_cover_every_arm() -> None:
    assert _LAYOUTS["hand_pose"].action_dim == 57
    assert _LAYOUTS["hand_pose_camera_free"].action_dim == 48
    assert _LAYOUTS["body_head_wrists"].action_dim == 57
    assert HumanPoseLayout.hand_pose(rotation_format="rot9d").action_dim == 3 * 12 + 2 * 15
    for layout in _LAYOUTS.values():
        slices = [s for s in (layout.camera_slice, layout.head_slice) if s is not None]
        for hand in range(2):
            slices += [layout.wrist_slice(hand), layout.fingers_slice(hand)]
        starts = [s.start for s in slices]
        assert starts == sorted(starts) and slices[0].start == 0 and slices[-1].stop == layout.action_dim
        assert all(a.stop == b.start for a, b in zip(slices, slices[1:]))
    assert _LAYOUTS["hand_pose"].head_slice is None and _LAYOUTS["body_head_wrists"].camera_slice is None
    assert _LAYOUTS["body_head_wrists"].head_slice == slice(0, 9)
    with pytest.raises(ValueError):
        HumanPoseLayout.hand_pose(rotation_format="quat_xyzw")


@pytest.mark.L0
def test_hand_layout_is_inferred_from_the_raw_width_like_the_eval() -> None:
    assert HumanPoseLayout.from_hand_action_dim(57).has_camera is True
    assert HumanPoseLayout.from_hand_action_dim(64).has_camera is True  # padded model output
    assert HumanPoseLayout.from_hand_action_dim(48).has_camera is False
    assert HumanPoseLayout.from_hand_action_dim(50).has_camera is False
    with pytest.raises(ValueError):
        HumanPoseLayout.from_hand_action_dim(40)


@pytest.mark.L0
@pytest.mark.parametrize("name", sorted(_LAYOUTS))
def test_initial_state_row_round_trips_through_encode_decode(name: str) -> None:
    layout = _LAYOUTS[name]
    rng = np.random.default_rng(hash(name) % 1000)
    anchors = _anchors(rng, layout)
    a0 = encode_initial_state_row(layout, anchors)
    assert a0.shape == (layout.action_dim,) and a0.dtype == np.float32
    decoded = decode_initial_state_row(a0, layout)
    _assert_anchors_close(decoded, anchors)
    if layout.has_camera:
        np.testing.assert_allclose(decoded.camera_pose, np.eye(4), atol=1e-6)  # a0 carries no ego-motion
    # Model outputs are padded to max_action_dim: the tail is ignored.
    padded = np.concatenate([a0, np.full(64 - a0.shape[0], 7.0, dtype=np.float32)])
    _assert_anchors_close(decode_initial_state_row(padded, layout), anchors)
    with pytest.raises(ValueError):
        decode_initial_state_row(a0[:-1], layout)


@pytest.mark.L0
def test_camera_free_row_is_the_camera_row_without_its_first_block() -> None:
    """webhumanaction_hand drops the camera block; a0 decodes to the same anchors from either width."""
    rng = np.random.default_rng(11)
    anchors = _anchors(rng, _LAYOUTS["hand_pose"])
    a0_57 = encode_initial_state_row(_LAYOUTS["hand_pose"], anchors)
    a0_48 = encode_initial_state_row(_LAYOUTS["hand_pose_camera_free"], anchors)
    np.testing.assert_array_equal(a0_57[9:], a0_48)
    _assert_anchors_close(decode_initial_state_row(a0_48, _LAYOUTS["hand_pose_camera_free"]), anchors)


@pytest.mark.L0
def test_body_row_layout_matches_the_eval_contract() -> None:
    """Body a0 = [head | R wrist | R tips | L wrist | L tips], every pose absolute in the camera frame."""
    layout = _LAYOUTS["body_head_wrists"]
    rng = np.random.default_rng(4)
    anchors = _anchors(rng, layout)
    a0 = encode_initial_state_row(layout, anchors)
    np.testing.assert_allclose(a0[0:3], anchors.head_pose[:3, 3], atol=1e-6)
    np.testing.assert_allclose(a0[9:12], anchors.wrist_poses[0][:3, 3], atol=1e-6)
    np.testing.assert_allclose(a0[18:33], anchors.fingers_local[0].reshape(-1), atol=1e-6)
    np.testing.assert_allclose(a0[33:36], anchors.wrist_poses[1][:3, 3], atol=1e-6)
    shifted = a0.copy()
    shifted[33:36] += [0.03, 0.0, 0.04]
    decoded = decode_initial_state_row(shifted, layout)
    np.testing.assert_allclose(
        np.linalg.norm(decoded.wrist_poses[1][:3, 3] - anchors.wrist_poses[1][:3, 3]), 0.05, atol=1e-6
    )
    np.testing.assert_allclose(decoded.wrist_poses[0], anchors.wrist_poses[0], atol=1e-5)


@pytest.mark.L0
@pytest.mark.parametrize("pose_convention", ["backward_framewise", "backward_anchored", "backward_chunk_anchored_16f"])
@pytest.mark.parametrize("name", sorted(_LAYOUTS))
def test_chains_reproduce_the_scene_with_a_static_camera(name: str, pose_convention: str) -> None:
    """a0 as the ONLY anchor + the delta rows reproduce camera-frame head / wrists / fingertips at every frame."""
    layout = _LAYOUTS[name]
    scene = _Scene(layout, pose_convention, num_steps=40, moving_camera=False, seed=5)
    a0, rows = split_initial_state(scene.window)
    chains = decode_human_pose_chains(
        rows, layout, decode_initial_state_row(a0, layout), pose_convention=pose_convention
    )
    assert chains.num_frames == 41
    for hand in range(2):
        np.testing.assert_allclose(chains.wrist_poses[hand], scene.wrists_cam[hand], atol=1e-4)
        np.testing.assert_allclose(chains.fingertips[hand], scene.tips_cam[hand], atol=1e-4)
    if layout.has_head:
        np.testing.assert_allclose(chains.head_poses, scene.head_cam, atol=1e-4)
    else:
        assert chains.head_poses is None
    if layout.has_camera:
        np.testing.assert_allclose(chains.camera_poses, np.broadcast_to(np.eye(4), (41, 4, 4)), atol=1e-4)
        assert chains.in_per_frame_camera() is not chains
    else:
        assert chains.camera_poses is None and chains.in_per_frame_camera() is chains


@pytest.mark.L0
@pytest.mark.parametrize("pose_convention", ["backward_anchored", "backward_chunk_anchored_16f"])
def test_moving_camera_chains_live_in_the_frame0_camera_frame_until_re_expressed(pose_convention: str) -> None:
    """Mecka-style egocentric data: rows are world deltas, anchors camera-frame. The decoder returns the
    frame-0 camera frame (residual ego-motion kept); in_per_frame_camera() recovers per-frame camera poses."""
    layout = _LAYOUTS["hand_pose"]
    scene = _Scene(layout, pose_convention, num_steps=32, moving_camera=True, seed=9)
    a0, rows = split_initial_state(scene.window)
    chains = decode_human_pose_chains(
        rows, layout, decode_initial_state_row(a0, layout), pose_convention=pose_convention
    )
    np.testing.assert_allclose(
        chains.camera_poses, scene.expected_in_camera0(np.broadcast_to(np.eye(4), (33, 4, 4))), atol=1e-4
    )
    for hand in range(2):
        np.testing.assert_allclose(
            chains.wrist_poses[hand], scene.expected_in_camera0(scene.wrists_cam[hand]), atol=1e-4
        )
    per_frame = chains.in_per_frame_camera()
    np.testing.assert_allclose(per_frame.camera_poses, np.broadcast_to(np.eye(4), (33, 4, 4)), atol=1e-6)
    for hand in range(2):
        np.testing.assert_allclose(per_frame.wrist_poses[hand], scene.wrists_cam[hand], atol=1e-4)
        np.testing.assert_allclose(per_frame.fingertips[hand], scene.tips_cam[hand], atol=1e-4)
    # Placing the frame-0 camera in a world frame moves every chain rigidly.
    world = decode_human_pose_chains(
        rows,
        layout,
        decode_initial_state_row(a0, layout),
        pose_convention=pose_convention,
        camera_pose0=scene.camera_c2w[0],
    )
    np.testing.assert_allclose(world.camera_poses, scene.camera_c2w, atol=1e-4)
    np.testing.assert_allclose(world.wrist_poses[1], scene.camera_c2w @ scene.wrists_cam[1], atol=1e-4)


@pytest.mark.L0
def test_padded_rows_and_bad_shapes() -> None:
    layout = _LAYOUTS["hand_pose_camera_free"]
    scene = _Scene(layout, "backward_chunk_anchored_16f", num_steps=8, moving_camera=False, seed=1)
    a0, rows = split_initial_state(scene.window)
    anchors = decode_initial_state_row(a0, layout)
    padded_rows = np.concatenate([rows, np.zeros((rows.shape[0], 64 - rows.shape[1]), dtype=rows.dtype)], axis=-1)
    plain = decode_human_pose_chains(rows, layout, anchors, pose_convention="backward_chunk_anchored_16f")
    padded = decode_human_pose_chains(padded_rows, layout, anchors, pose_convention="backward_chunk_anchored_16f")
    np.testing.assert_array_equal(plain.wrist_poses[0], padded.wrist_poses[0])
    with pytest.raises(ValueError):
        decode_human_pose_chains(rows[:, :-1], layout, anchors, pose_convention="backward_chunk_anchored_16f")
    with pytest.raises(ValueError):
        decode_human_pose_chains(
            rows, _LAYOUTS["body_head_wrists"], anchors, pose_convention="backward_chunk_anchored_16f"
        )
    with pytest.raises(ValueError):
        split_initial_state(np.zeros((0, 48)))
