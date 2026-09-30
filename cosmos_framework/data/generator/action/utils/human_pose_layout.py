# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Human-pose action layouts and decoders (hand_pose / Mecka, webhumanaction_hand, webhumanaction_body).

The human action arms share one row grammar::

    [camera(3+rot)]? [head(3+rot)]? right_wrist(3+rot) right_fingers(J*3) left_wrist(3+rot) left_fingers(J*3)

* ``hand_pose`` (Mecka, 57D for rot6d / 5 fingertips): camera + two hands.
* ``webhumanaction_hand`` (Action100M hands, 48D): two hands, the static camera block dropped.
* ``webhumanaction_body`` (Action100M body, 57D): head + two hands, no camera block.

Rows ``1..T`` are relative-pose deltas (``pose_abs_to_rel``) for the camera / head / wrist blocks and
per-frame fingertip positions expressed in that frame's (aligned) wrist frame. Image2Action models
additionally predict row 0, the **initial-state row a0**: the same grammar, but every pose block is
the ABSOLUTE camera-frame pose at frame 0 (the camera block is the identity) and the finger blocks are
the frame-0 fingertips in the wrist-0 frame. a0 is exactly the anchor set ``pose_rel_to_abs`` needs to
turn the delta rows back into camera-frame trajectories, which is what this module does:

* :class:`HumanPoseLayout` -- the block arithmetic for the three arms;
* :func:`encode_initial_state_row` / :func:`decode_initial_state_row` -- a0 <-> :class:`HumanPoseAnchors`;
* :func:`decode_human_pose_chains` -- delta rows + anchors -> camera / head / wrist / fingertip
  trajectories in the frame-0 camera frame (:class:`HumanPoseChains`), with
  :meth:`HumanPoseChains.in_per_frame_camera` for moving-camera (egocentric) data.

Everything here is plain numpy on denormalized actions: run the dataset's action normalizer first.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from cosmos_framework.data.generator.action.utils.pose_utils import (
    PoseConvention,
    RotationConvention,
    absolute_pose_to_vector,
    convert_rotation,
    pose_rel_to_abs,
)

NUM_HAND_JOINTS = 21
WRIST_JOINT_IDX = 0
FINGERTIP_JOINT_IDXS: tuple[int, ...] = (4, 8, 12, 16, 20)
NUM_HANDS = 2  # layout order: right, left (every Cosmos human dataset)

_ROTATION_DIMS: dict[str, int] = {"rot6d": 6, "rot9d": 9}


@dataclass(frozen=True)
class HumanPoseLayout:
    """Block layout of one human-pose action row.

    Args:
        has_camera: Leading camera pose block (``hand_pose`` / Mecka).
        has_head: Head pose block, after the camera block if any (``webhumanaction_body``).
        rotation_format: Rotation block convention of every pose block (``rot6d`` or ``rot9d``).
        num_finger_joints: Non-wrist keypoints per hand carried as wrist-frame positions (5 = fingertips).
    """

    has_camera: bool
    has_head: bool
    rotation_format: RotationConvention = "rot6d"
    num_finger_joints: int = len(FINGERTIP_JOINT_IDXS)

    def __post_init__(self) -> None:
        if self.rotation_format not in _ROTATION_DIMS:
            raise ValueError(f"HumanPoseLayout supports rot6d / rot9d, got {self.rotation_format!r}")
        if self.num_finger_joints < 0:
            raise ValueError(f"num_finger_joints must be >= 0, got {self.num_finger_joints}")

    # -- presets ------------------------------------------------------------------------------
    @classmethod
    def hand_pose(cls, **kwargs: object) -> HumanPoseLayout:
        """``hand_pose`` / Mecka: camera + two hands (57D for rot6d, 5 fingertips)."""
        return cls(has_camera=True, has_head=False, **kwargs)  # type: ignore[arg-type]

    @classmethod
    def hand_pose_camera_free(cls, **kwargs: object) -> HumanPoseLayout:
        """``webhumanaction_hand``: two hands, no camera block (48D for rot6d, 5 fingertips)."""
        return cls(has_camera=False, has_head=False, **kwargs)  # type: ignore[arg-type]

    @classmethod
    def body_head_wrists(cls, **kwargs: object) -> HumanPoseLayout:
        """``webhumanaction_body``: head + two hands, no camera block (57D for rot6d, 5 fingertips)."""
        return cls(has_camera=False, has_head=True, **kwargs)  # type: ignore[arg-type]

    @classmethod
    def from_hand_action_dim(
        cls,
        action_dim: int,
        *,
        rotation_format: RotationConvention = "rot6d",
        num_finger_joints: int = len(FINGERTIP_JOINT_IDXS),
    ) -> HumanPoseLayout:
        """Infer the hand layout (camera-inclusive vs camera-free) from a raw action width.

        Mirrors the hand-pose eval convention: a width that fits the camera-inclusive layout is
        camera-inclusive (padding beyond the layout is tolerated), otherwise camera-free.
        Body rows cannot be told apart from Mecka rows by width; use :meth:`body_head_wrists`.
        """
        camera_free = cls(
            has_camera=False, has_head=False, rotation_format=rotation_format, num_finger_joints=num_finger_joints
        )
        with_camera = replace(camera_free, has_camera=True)
        if action_dim >= with_camera.action_dim:
            return with_camera
        if action_dim >= camera_free.action_dim:
            return camera_free
        raise ValueError(
            f"action_dim {action_dim} is narrower than the camera-free hand layout ({camera_free.action_dim})"
        )

    # -- arithmetic ---------------------------------------------------------------------------
    @property
    def rotation_dim(self) -> int:
        return _ROTATION_DIMS[self.rotation_format]

    @property
    def pose_dim(self) -> int:
        """Width of one pose block: translation(3) + rotation."""
        return 3 + self.rotation_dim

    @property
    def fingers_dim(self) -> int:
        return 3 * self.num_finger_joints

    @property
    def per_hand_dim(self) -> int:
        return self.pose_dim + self.fingers_dim

    @property
    def action_dim(self) -> int:
        return int(self.has_camera) * self.pose_dim + int(self.has_head) * self.pose_dim + NUM_HANDS * self.per_hand_dim

    @property
    def camera_slice(self) -> slice | None:
        return slice(0, self.pose_dim) if self.has_camera else None

    @property
    def head_slice(self) -> slice | None:
        if not self.has_head:
            return None
        start = int(self.has_camera) * self.pose_dim
        return slice(start, start + self.pose_dim)

    def _hand_start(self, hand: int) -> int:
        if hand not in range(NUM_HANDS):
            raise IndexError(f"hand must be 0 (right) or 1 (left), got {hand}")
        return (int(self.has_camera) + int(self.has_head)) * self.pose_dim + hand * self.per_hand_dim

    def wrist_slice(self, hand: int) -> slice:
        start = self._hand_start(hand)
        return slice(start, start + self.pose_dim)

    def fingers_slice(self, hand: int) -> slice:
        start = self._hand_start(hand) + self.pose_dim
        return slice(start, start + self.fingers_dim)


@dataclass(frozen=True)
class HumanPoseAnchors:
    """Frame-0 anchors of a human-pose window, in the frame-0 camera frame.

    ``wrist_poses`` / ``fingers_local`` are per hand in layout order (right, left): ``(4, 4)`` aligned
    wrist poses and ``(J, 3)`` fingertips in each wrist-0 frame. ``head_pose`` is the ``(4, 4)`` head
    pose (body layouts), ``camera_pose`` the camera block of an a0 row (identity by construction).
    """

    wrist_poses: tuple[np.ndarray, np.ndarray]
    fingers_local: tuple[np.ndarray, np.ndarray]
    head_pose: np.ndarray | None = None
    camera_pose: np.ndarray | None = None


@dataclass(frozen=True)
class HumanPoseChains:
    """Decoded trajectories, ``T + 1`` poses each (frame 0 = the anchors), in the frame-0 camera frame.

    ``camera_poses`` is ``None`` for layouts without a camera block (static camera: every frame's
    camera frame is the frame-0 camera frame). ``fingertips`` are camera-frame positions ``(T+1, J, 3)``.
    """

    wrist_poses: tuple[np.ndarray, np.ndarray]
    fingertips: tuple[np.ndarray, np.ndarray]
    head_poses: np.ndarray | None = None
    camera_poses: np.ndarray | None = None

    @property
    def num_frames(self) -> int:
        return int(self.wrist_poses[0].shape[0])

    def transformed_by(self, transform: np.ndarray) -> HumanPoseChains:
        """Left-multiply a ``(4, 4)`` or per-frame ``(T+1, 4, 4)`` rigid transform onto every chain."""
        transform = np.asarray(transform, dtype=np.float64)
        if transform.ndim == 2:
            transform = np.broadcast_to(transform, (self.num_frames, 4, 4))
        rot, trans = transform[:, None, :3, :3], transform[:, None, :3, 3]  # [T+1,1,3,3], [T+1,1,3]

        def _points(points: np.ndarray) -> np.ndarray:  # [T+1,J,3] -> [T+1,J,3]
            return np.einsum("tnij,tnj->tni", np.broadcast_to(rot, (*points.shape[:2], 3, 3)), points) + trans

        return HumanPoseChains(
            wrist_poses=(transform @ self.wrist_poses[0], transform @ self.wrist_poses[1]),
            fingertips=(_points(self.fingertips[0]), _points(self.fingertips[1])),
            head_poses=None if self.head_poses is None else transform @ self.head_poses,
            camera_poses=None if self.camera_poses is None else transform @ self.camera_poses,
        )

    def in_per_frame_camera(self) -> HumanPoseChains:
        """Re-express head / wrists / fingertips in each frame's own camera frame.

        For a moving (egocentric) camera the delta rows are integrated in the frame-0 camera frame, so
        frame ``t`` still carries the residual ego-motion ``E[t] = C0^-1 C_t``; left-multiplying
        ``E[t]^-1`` returns true per-frame camera-frame poses (frame 0 is unchanged). Without a camera
        chain the camera is static and this is the identity.
        """
        if self.camera_poses is None:
            return self
        return self.transformed_by(np.linalg.inv(self.camera_poses))


def _pose_from_block(block: np.ndarray, rotation_format: RotationConvention) -> np.ndarray:
    """``[t(3), rot(...)]`` -> ``(4, 4)`` float64 pose; the rotation is projected onto SO(3)."""
    block = np.asarray(block, dtype=np.float64).reshape(-1)
    rotation = np.asarray(
        convert_rotation(block[3:].astype(np.float32), rotation_format, "matrix", normalize_matrix=True),
        dtype=np.float64,
    ).reshape(3, 3)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = block[:3]
    return pose


def _check_width(array: np.ndarray, layout: HumanPoseLayout, what: str) -> np.ndarray:
    if array.shape[-1] < layout.action_dim:
        raise ValueError(f"{what} has {array.shape[-1]} dims, the layout needs {layout.action_dim}")
    return array[..., : layout.action_dim]


def split_initial_state(action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a ``(T+1, D)`` Image2Action window into ``(a0 (D,), delta rows (T, D))``."""
    action = np.asarray(action)
    if action.ndim != 2 or action.shape[0] < 1:
        raise ValueError(f"expected a (T+1, D) action window, got shape {action.shape}")
    return action[0], action[1:]


def encode_initial_state_row(layout: HumanPoseLayout, anchors: HumanPoseAnchors) -> np.ndarray:
    """Build the initial-state row a0 ``(D,)`` float32 from frame-0 anchors.

    Pose blocks are ``absolute_pose_to_vector`` of the camera-frame frame-0 poses (camera block: identity
    unless ``anchors.camera_pose`` is given), finger blocks the wrist-0-frame fingertips flattened.
    """
    blocks: list[np.ndarray] = []
    if layout.has_camera:
        camera_pose = np.eye(4, dtype=np.float32) if anchors.camera_pose is None else anchors.camera_pose
        blocks.append(absolute_pose_to_vector(np.asarray(camera_pose), layout.rotation_format))
    if layout.has_head:
        if anchors.head_pose is None:
            raise ValueError("layout has a head block but anchors.head_pose is None")
        blocks.append(absolute_pose_to_vector(np.asarray(anchors.head_pose), layout.rotation_format))
    for hand in range(NUM_HANDS):
        fingers = np.asarray(anchors.fingers_local[hand], dtype=np.float32).reshape(-1)
        if fingers.shape[0] != layout.fingers_dim:
            raise ValueError(
                f"hand {hand}: expected {layout.num_finger_joints} x 3 fingertips, got {fingers.shape[0]} values"
            )
        blocks.append(absolute_pose_to_vector(np.asarray(anchors.wrist_poses[hand]), layout.rotation_format))
        blocks.append(fingers)
    row = np.concatenate(blocks).astype(np.float32)
    assert row.shape == (layout.action_dim,), (row.shape, layout.action_dim)
    return row


def decode_initial_state_row(a0: np.ndarray, layout: HumanPoseLayout) -> HumanPoseAnchors:
    """Decode an initial-state row (width ``>= layout.action_dim``; padding is ignored) into anchors."""
    a0 = _check_width(np.asarray(a0, dtype=np.float64).reshape(-1), layout, "initial-state row")
    camera_pose = (
        None if layout.camera_slice is None else _pose_from_block(a0[layout.camera_slice], layout.rotation_format)
    )
    head_pose = None if layout.head_slice is None else _pose_from_block(a0[layout.head_slice], layout.rotation_format)
    wrist_poses = tuple(
        _pose_from_block(a0[layout.wrist_slice(hand)], layout.rotation_format) for hand in range(NUM_HANDS)
    )
    fingers_local = tuple(
        a0[layout.fingers_slice(hand)].reshape(layout.num_finger_joints, 3).copy() for hand in range(NUM_HANDS)
    )
    return HumanPoseAnchors(
        wrist_poses=wrist_poses,  # type: ignore[arg-type]
        fingers_local=fingers_local,  # type: ignore[arg-type]
        head_pose=head_pose,
        camera_pose=camera_pose,
    )


def decode_human_pose_chains(
    action_rows: np.ndarray,
    layout: HumanPoseLayout,
    anchors: HumanPoseAnchors,
    *,
    pose_convention: PoseConvention,
    camera_pose0: np.ndarray | None = None,
) -> HumanPoseChains:
    """Integrate ``T`` denormalized delta rows from frame-0 anchors into camera-frame trajectories.

    Args:
        action_rows: ``(T, D)`` delta rows (row 0 of an Image2Action window removed, see
            :func:`split_initial_state`); ``D >= layout.action_dim``, padding ignored.
        layout: Row layout.
        anchors: Frame-0 anchors -- decoded from a predicted / GT a0 row, or built from the dataset's
            frame-0 camera-frame poses for models without a0.
        pose_convention: Convention the rows were encoded with (``backward_chunk_anchored_16f`` for the
            HumanAction recipe); must match the dataset.
        camera_pose0: Optional ``(4, 4)`` pose of the frame-0 camera in an external (world) frame. Every
            chain is left-multiplied by it, so the output lives in that frame; default identity, i.e.
            every trajectory is expressed in the frame-0 camera frame.

    Returns:
        :class:`HumanPoseChains` with ``T + 1`` poses per chain.
    """
    rows = _check_width(np.asarray(action_rows, dtype=np.float64), layout, "action rows")
    if rows.ndim != 2:
        raise ValueError(f"expected (T, D) action rows, got shape {rows.shape}")
    num_steps = rows.shape[0]
    rotation_format = layout.rotation_format

    def _chain(block_slice: slice, initial_pose: np.ndarray) -> np.ndarray:
        return pose_rel_to_abs(
            rows[:, block_slice],
            rotation_format=rotation_format,
            pose_convention=pose_convention,
            initial_pose=np.asarray(initial_pose, dtype=np.float64),
        )  # [T+1,4,4]

    camera_poses = None
    if layout.camera_slice is not None:
        camera_poses = _chain(layout.camera_slice, np.eye(4))  # frame-0 camera frame; placed below
    head_poses = None
    if layout.head_slice is not None:
        if anchors.head_pose is None:
            raise ValueError("layout has a head block but anchors.head_pose is None")
        head_poses = _chain(layout.head_slice, anchors.head_pose)

    wrist_poses: list[np.ndarray] = []
    fingertips: list[np.ndarray] = []
    for hand in range(NUM_HANDS):
        poses = _chain(layout.wrist_slice(hand), anchors.wrist_poses[hand])  # [T+1,4,4]
        local0 = np.asarray(anchors.fingers_local[hand], dtype=np.float64).reshape(1, layout.num_finger_joints, 3)
        local = rows[:, layout.fingers_slice(hand)].reshape(num_steps, layout.num_finger_joints, 3)  # [T,J,3]
        local_all = np.concatenate([local0, local], axis=0)  # [T+1,J,3]
        tips = np.einsum("tij,tnj->tni", poses[:, :3, :3], local_all) + poses[:, None, :3, 3]  # [T+1,J,3]
        wrist_poses.append(poses)
        fingertips.append(tips)
    chains = HumanPoseChains(
        wrist_poses=(wrist_poses[0], wrist_poses[1]),
        fingertips=(fingertips[0], fingertips[1]),
        head_poses=head_poses,
        camera_poses=camera_poses,
    )
    return chains if camera_pose0 is None else chains.transformed_by(np.asarray(camera_pose0, dtype=np.float64))


__all__ = [
    "FINGERTIP_JOINT_IDXS",
    "NUM_HAND_JOINTS",
    "NUM_HANDS",
    "WRIST_JOINT_IDX",
    "HumanPoseAnchors",
    "HumanPoseChains",
    "HumanPoseLayout",
    "decode_human_pose_chains",
    "decode_initial_state_row",
    "encode_initial_state_row",
    "split_initial_state",
]
