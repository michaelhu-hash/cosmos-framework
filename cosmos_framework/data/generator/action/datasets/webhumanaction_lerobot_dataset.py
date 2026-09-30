# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Action100M (WebHumanAction) human hand and body datasets in LeRobot v3 format.

These are the two Action100M arms of Cosmos3-Nano-HumanAction:

* :class:`WebHumanActionHandLeRobotDataset` -- domain ``webhumanaction_hand`` (id 31), 48D
  ``[right_wrist(9), right_fingertips(15), left_wrist(9), left_fingertips(15)]``. Same hand chain as the
  camera-inclusive ``hand_pose`` layout, with the leading camera block dropped (the Action100M camera is static).
* :class:`WebHumanActionBodyLeRobotDataset` -- domain ``webhumanaction_body`` (id 24), 57D
  ``[head(9), right_wrist(9), right_fingertips(15), left_wrist(9), left_fingertips(15)]``. The head pose replaces
  the camera block; wrists and head are lifted into the world frame through the per-frame camera pose.

Both use the HumanAction post-training recipe by default: 15 fps, 72-step chunks (73 frames),
``backward_chunk_anchored_16f`` pose deltas (the anchor resets every 16 steps), ``rot6d`` rotations,
``piecewise_asinh_rot`` normalization with per-domain stats under ``normalizer_stats/``. Wrist poses use the
Action100M per-hand frame alignment (right/left differ), fingertips are the five tip joints expressed in the
aligned per-frame wrist frame. Expected LeRobot features are the Action100M export keys
(``observation.state.hand_{left,right}_cam[_rotation]``, ``observation.state.camera_{position,rotation}`` and,
for body, ``observation.state.head_cam[_rotation]``), 21 joints per hand, xyzw quaternions.

**Image2Action (a0).** With ``initial_state="predict"`` the inverse-dynamics / wam windows carry ``T + 1`` rows:
row 0 is the frame-0 initial-state row a0 -- the same block grammar, but every pose block is the ABSOLUTE
camera-frame pose at frame 0 (aligned wrists; the head for the body arm) and the finger blocks are the frame-0
fingertips in the wrist-0 frame (see ``human_pose_layout``). The item is flagged ``has_initial_state`` so the
transform generates row 0 instead of conditioning on it, and row 0 is normalized with the dedicated
``*_initial_state_stats.json``. Forward-dynamics windows never carry a0. Decode a predicted window with
``split_initial_state`` + ``decode_initial_state_row`` + ``decode_human_pose_chains`` after :meth:`denormalize`.
"""

from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow.parquet as pq
import torch
from lerobot.datasets.video_utils import decode_video_frames

from cosmos_framework.data.generator.action.datasets.base_dataset import ActionBaseDataset
from cosmos_framework.data.generator.action.utils.action_spec import ActionSpec, Pos, Rot, build_action_spec
from cosmos_framework.data.generator.action.utils.human_pose_layout import (
    HumanPoseAnchors,
    HumanPoseLayout,
    encode_initial_state_row,
)
from cosmos_framework.data.generator.action.utils.pose_utils import (
    PoseConvention,
    RotationConvention,
    build_abs_pose_from_components,
    pose_abs_to_rel,
)

Viewpoint = Literal["ego_view"]

HAND_RIGHT_POSITION_KEY = "observation.state.hand_right_cam"
HAND_RIGHT_ROTATION_KEY = "observation.state.hand_right_cam_rotation"
HAND_LEFT_POSITION_KEY = "observation.state.hand_left_cam"
HAND_LEFT_ROTATION_KEY = "observation.state.hand_left_cam_rotation"
CAMERA_POSITION_KEY = "observation.state.camera_position"
CAMERA_ROTATION_KEY = "observation.state.camera_rotation"
HEAD_POSITION_KEY = "observation.state.head_cam"
HEAD_ROTATION_KEY = "observation.state.head_cam_rotation"
IMAGE_FEATURE = "observation.images.main"

NUM_JOINTS = 21
QUAT_DIM = 4
WRIST_JOINT_IDX = 0
FINGERTIP_JOINT_IDXS = (4, 8, 12, 16, 20)
POSE_DIM = 9  # translation(3) + rot6d(6)
FINGERTIP_DIM = 3 * len(FINGERTIP_JOINT_IDXS)
HAND_ACTION_DIM = 2 * (POSE_DIM + FINGERTIP_DIM)  # 48
BODY_ACTION_DIM = POSE_DIM + HAND_ACTION_DIM  # 57

DEFAULT_FPS = 15.0
DEFAULT_CHUNK_LENGTH = 72
DEFAULT_POSE_CONVENTION: PoseConvention = "backward_chunk_anchored_16f"
DEFAULT_ROTATION_FORMAT: RotationConvention = "rot6d"
DEFAULT_ACTION_NORMALIZATION = "piecewise_asinh_rot"

_STATS_DIR = Path(__file__).parent.parent / "normalizer_stats"
HAND_NORMALIZER_PATH = _STATS_DIR / "webhumanaction_hand_lerobot_stats.json"
BODY_NORMALIZER_PATH = _STATS_DIR / "webhumanaction_body_lerobot_stats.json"
# Image2Action initial-state (a0) stats of the HumanAction joint a0 model (q01/q99).
HAND_INITIAL_STATE_NORMALIZER_PATH = _STATS_DIR / "webhumanaction_hand_lerobot_initial_state_stats.json"
BODY_INITIAL_STATE_NORMALIZER_PATH = _STATS_DIR / "webhumanaction_body_lerobot_initial_state_stats.json"
HAND_LAYOUT = HumanPoseLayout.hand_pose_camera_free()
BODY_LAYOUT = HumanPoseLayout.body_head_wrists()

# Pure rotations taking the Action100M wrist joint frames into the unified convention
# (X = thumb-to-pinky, Y = outward palm normal, Z = wrist-to-fingertips). Left and right differ.
WRIST_FRAME_ALIGN_ACTION100M_RIGHT = np.array(
    [[0, 0, -1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
    dtype=np.float32,
)
WRIST_FRAME_ALIGN_ACTION100M_LEFT = np.array(
    [[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
    dtype=np.float32,
)


def _as_float32(values: Any, width: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"Expected {name!r} with shape [T+1, {width}], got {array.shape}.")
    return array


def wrist_poses_in_camera(positions: np.ndarray, rotations: np.ndarray, alignment: np.ndarray) -> np.ndarray:
    """Aligned wrist SE(3) poses in the camera frame, ``[T+1, 4, 4]``.

    ``positions`` are the 21 joint positions ``[T+1, 63]`` (wrist first), ``rotations`` the 21 xyzw joint
    quaternions ``[T+1, 84]``. The wrist joint pose is right-multiplied by the pure-rotation ``alignment``.
    """
    wrist_position = positions[:, WRIST_JOINT_IDX * 3 : WRIST_JOINT_IDX * 3 + 3]  # [T+1,3]
    wrist_quat = rotations.reshape(positions.shape[0], NUM_JOINTS, QUAT_DIM)[:, WRIST_JOINT_IDX]  # [T+1,4]
    return build_abs_pose_from_components(wrist_position, wrist_quat, "quat_xyzw") @ alignment  # [T+1,4,4]


def fingertips_in_wrist_frame(
    positions: np.ndarray, wrist_poses: np.ndarray, frames: slice = slice(1, None)
) -> np.ndarray:
    """Five fingertip positions expressed in the aligned wrist frame of the same frame, ``[N, 15]``.

    Defaults to frames ``1..T`` so the block lines up with the relative-pose rows, which spend frame 0 as the first
    anchor; ``frames=slice(0, 1)`` gives the frame-0 fingertips of the initial-state row a0.
    """
    selected = positions[frames].reshape(-1, NUM_JOINTS, 3)  # [N,21,3]
    tips = selected[:, FINGERTIP_JOINT_IDXS, :]  # [N,5,3]
    tips_h = np.concatenate([tips, np.ones((*tips.shape[:-1], 1), dtype=np.float32)], axis=-1)  # [N,5,4]
    wrist_inv = np.linalg.inv(wrist_poses[frames])  # [N,4,4]
    tips_wrist = np.einsum("tij,tnj->tni", wrist_inv, tips_h)[..., :3]  # [N,5,3]
    return tips_wrist.reshape(len(selected), -1).astype(np.float32)  # [N,15]


def _hand_blocks(
    sample: dict[str, Any],
    *,
    pose_convention: PoseConvention,
    rotation_format: RotationConvention,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, HumanPoseAnchors]:
    """Shared Action100M hand chain: ``(camera_c2w [T+1,4,4], right_hand [T,24], left_hand [T,24], frame-0 anchors)``.

    The anchors are the aligned camera-frame wrist poses at frame 0 and the frame-0 fingertips in the wrist-0 frame:
    the wrist / finger part of the initial-state row a0 (the body arm adds the head; see ``human_pose_layout``).
    """
    camera_c2w = build_abs_pose_from_components(
        _as_float32(sample[CAMERA_POSITION_KEY], 3, CAMERA_POSITION_KEY),
        _as_float32(sample[CAMERA_ROTATION_KEY], 4, CAMERA_ROTATION_KEY),
        "quat_xyzw",
    )  # [T+1,4,4]
    hands = []
    wrist0: list[np.ndarray] = []
    fingers0: list[np.ndarray] = []
    for position_key, rotation_key, alignment in (
        (HAND_RIGHT_POSITION_KEY, HAND_RIGHT_ROTATION_KEY, WRIST_FRAME_ALIGN_ACTION100M_RIGHT),
        (HAND_LEFT_POSITION_KEY, HAND_LEFT_ROTATION_KEY, WRIST_FRAME_ALIGN_ACTION100M_LEFT),
    ):
        positions = _as_float32(sample[position_key], NUM_JOINTS * 3, position_key)
        rotations = _as_float32(sample[rotation_key], NUM_JOINTS * QUAT_DIM, rotation_key)
        wrist_camera = wrist_poses_in_camera(positions, rotations, alignment)  # [T+1,4,4]
        wrist_world = camera_c2w @ wrist_camera  # [T+1,4,4]
        wrist_rel = pose_abs_to_rel(wrist_world, rotation_format=rotation_format, pose_convention=pose_convention)
        hands.append(np.concatenate([wrist_rel, fingertips_in_wrist_frame(positions, wrist_camera)], axis=-1))
        wrist0.append(wrist_camera[0].astype(np.float64))
        fingers0.append(fingertips_in_wrist_frame(positions, wrist_camera, frames=slice(0, 1)).reshape(-1, 3))
    anchors = HumanPoseAnchors(wrist_poses=(wrist0[0], wrist0[1]), fingers_local=(fingers0[0], fingers0[1]))
    return camera_c2w, hands[0], hands[1], anchors


def build_webhumanaction_hand_action(
    sample: dict[str, Any],
    *,
    pose_convention: PoseConvention = DEFAULT_POSE_CONVENTION,
    rotation_format: RotationConvention = DEFAULT_ROTATION_FORMAT,
    include_initial_state: bool = False,
) -> np.ndarray:
    """48D hand action ``[right_wrist, right_fingertips, left_wrist, left_fingertips]`` from ``T+1`` frames.

    ``include_initial_state`` prepends the frame-0 initial-state row a0 (``[T+1, 48]``).
    """
    _, right_hand, left_hand, anchors = _hand_blocks(
        sample, pose_convention=pose_convention, rotation_format=rotation_format
    )
    action = np.concatenate([right_hand, left_hand], axis=-1).astype(np.float32)  # [T,48]
    if include_initial_state:
        a0 = encode_initial_state_row(HumanPoseLayout.hand_pose_camera_free(rotation_format=rotation_format), anchors)
        action = np.concatenate([a0[None], action], axis=0)  # [T+1,48]
    return action


def build_webhumanaction_body_action(
    sample: dict[str, Any],
    *,
    pose_convention: PoseConvention = DEFAULT_POSE_CONVENTION,
    rotation_format: RotationConvention = DEFAULT_ROTATION_FORMAT,
    include_initial_state: bool = False,
) -> np.ndarray:
    """57D body action ``[head, right_wrist, right_fingertips, left_wrist, left_fingertips]`` from ``T+1`` frames.

    ``include_initial_state`` prepends the frame-0 initial-state row a0 (``[T+1, 57]``, head block = the absolute
    camera-frame head pose at frame 0).
    """
    camera_c2w, right_hand, left_hand, anchors = _hand_blocks(
        sample, pose_convention=pose_convention, rotation_format=rotation_format
    )
    head_camera = build_abs_pose_from_components(
        _as_float32(sample[HEAD_POSITION_KEY], 3, HEAD_POSITION_KEY),
        _as_float32(sample[HEAD_ROTATION_KEY], 4, HEAD_ROTATION_KEY),
        "quat_xyzw",
    )  # [T+1,4,4]
    head_rel = pose_abs_to_rel(
        camera_c2w @ head_camera, rotation_format=rotation_format, pose_convention=pose_convention
    )  # [T,9]
    action = np.concatenate([head_rel, right_hand, left_hand], axis=-1).astype(np.float32)  # [T,57]
    if include_initial_state:
        layout = HumanPoseLayout.body_head_wrists(rotation_format=rotation_format)
        a0 = encode_initial_state_row(layout, replace(anchors, head_pose=head_camera[0].astype(np.float64)))
        action = np.concatenate([a0[None], action], axis=0)  # [T+1,57]
    return action


class _WebHumanActionLeRobotDataset(ActionBaseDataset):
    """Window sampling, video decoding and captions shared by the hand and body arms."""

    _DOMAIN_NAME: str
    _ACTION_CAPTION_DATASET_NAME: str  # ACTION_CAPTION_ATTRIBUTE_ADAPTER protocol (released-recipe prompt)
    _ACTION_DIM: int
    _NORMALIZER_PATH: Path
    _INITIAL_STATE_NORMALIZER_PATH: Path

    def __init__(
        self,
        root: str,
        fps: float = DEFAULT_FPS,
        chunk_length: int = DEFAULT_CHUNK_LENGTH,
        mode: str = "forward_dynamics",
        pose_convention: PoseConvention = DEFAULT_POSE_CONVENTION,
        rotation_format: RotationConvention = DEFAULT_ROTATION_FORMAT,
        tolerance_s: float = 2e-4,
        viewpoint: Viewpoint = "ego_view",
        action_normalization: str | None = DEFAULT_ACTION_NORMALIZATION,
        sample_stride: int = 1,
        image_key: str = IMAGE_FEATURE,
        stats_path: str | Path | None = None,
        initial_state: Literal["predict"] | None = None,
        initial_state_stats_path: str | Path | None = None,
        split: str = "full",
        val_ratio: float = 0.0,
        split_seed: int = 42,
        snap_to_subtask: bool = False,
        caption_semantics: bool = True,
    ) -> None:
        if viewpoint != "ego_view":
            raise NotImplementedError("WebHumanAction data only supports ego_view.")
        if rotation_format != "rot6d":
            raise NotImplementedError("WebHumanAction actions use rot6d rotations.")
        super().__init__(
            root=root,
            domain_name=self._DOMAIN_NAME,
            fps=fps,
            chunk_length=chunk_length,
            mode=mode,
            pose_convention=pose_convention,
            tolerance_s=tolerance_s,
            viewpoint=viewpoint,
            action_normalization=action_normalization,
            sample_stride=sample_stride,
            stats_path=stats_path,
            initial_state=initial_state,
            initial_state_stats_path=initial_state_stats_path,
            split=split,
            val_ratio=val_ratio,
            split_seed=split_seed,
            snap_to_subtask=snap_to_subtask,
            action_caption_dataset_name=self._ACTION_CAPTION_DATASET_NAME if caption_semantics else None,
        )
        self._rotation_format: RotationConvention = rotation_format
        source_fps = float(self._info["fps"])
        source_stride = source_fps / self._fps
        if not source_stride.is_integer():
            raise ValueError(f"Source FPS {source_fps} must be an integer multiple of target FPS {self._fps}.")
        self._image_key = image_key
        # Window index: dense sliding windows, or one variable-length window per subtask (snap_to_subtask),
        # restricted to the train / val split. See ActionBaseDataset._init_window_index.
        self._init_window_index(source_stride=int(source_stride))
        subtasks_path = self._root / "meta" / "subtasks.parquet"
        self._subtasks = (
            {int(row["subtask_index"]): str(row["subtask"]) for row in pq.read_table(subtasks_path).to_pylist()}
            if subtasks_path.exists()
            else {}
        )

    @property
    def action_dim(self) -> int:
        return self._ACTION_DIM

    @property
    def rotation_format(self) -> RotationConvention:
        return self._rotation_format

    @classmethod
    def _stats_path(cls) -> Path:
        return cls._NORMALIZER_PATH

    @classmethod
    def _initial_state_stats_path(cls) -> Path:
        return cls._INITIAL_STATE_NORMALIZER_PATH

    def __len__(self) -> int:
        return len(self._windows or [])

    def _caption(self, rows: list[dict[str, Any]]) -> str:
        subtask_index = int(rows[0].get("subtask_index", -1))
        task = self._tasks[int(rows[0]["task_index"])]
        caption = self._subtasks.get(subtask_index, task)
        return random.choice([part.strip() for part in caption.split(" | ") if part.strip()] or [caption])

    def _load_video(self, episode: dict[str, Any], rows: list[dict[str, Any]]) -> torch.Tensor:
        timestamps = [float(row["timestamp"]) for row in rows]
        from_timestamp = float(episode.get(f"videos/{self._image_key}/from_timestamp", 0.0))
        return decode_video_frames(
            self._video_path(episode, self._image_key),
            [from_timestamp + timestamp for timestamp in timestamps],
            self._tolerance_s,
        )

    @staticmethod
    def _sample_from_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, np.ndarray]:
        return {key: np.asarray([row[key] for row in rows], dtype=np.float32) for key in keys}

    def _build_raw_action(self, rows: list[dict[str, Any]], include_initial_state: bool = False) -> torch.Tensor:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode = self._choose_mode()
        rows = self._window_rows(idx)
        episode = self._episodes[int(rows[0]["episode_index"])]
        video = self._load_video(episode, rows)
        include_initial_state = self.wants_initial_state(mode)
        raw_action = self._build_raw_action(rows, include_initial_state=include_initial_state)
        expected_rows = len(rows) - 1 + int(include_initial_state)  # snapped windows can be shorter than chunk_length
        if raw_action.shape != (expected_rows, self.action_dim):
            raise ValueError(
                f"Expected action shape {(expected_rows, self.action_dim)}, got {tuple(raw_action.shape)}."
            )
        return self._build_result(
            mode=mode,
            video=video,
            action=raw_action,
            ai_caption=self._caption(rows),
            has_initial_state=include_initial_state,
        )


class WebHumanActionHandLeRobotDataset(_WebHumanActionLeRobotDataset):
    """Action100M hands: domain ``webhumanaction_hand`` (31), 48D camera-free action."""

    _DOMAIN_NAME = "webhumanaction_hand"
    _ACTION_CAPTION_DATASET_NAME = "web_human_action_hand"
    _ACTION_DIM = HAND_ACTION_DIM
    _NORMALIZER_PATH = HAND_NORMALIZER_PATH
    _INITIAL_STATE_NORMALIZER_PATH = HAND_INITIAL_STATE_NORMALIZER_PATH
    _KEYS = (
        CAMERA_POSITION_KEY,
        CAMERA_ROTATION_KEY,
        HAND_RIGHT_POSITION_KEY,
        HAND_RIGHT_ROTATION_KEY,
        HAND_LEFT_POSITION_KEY,
        HAND_LEFT_ROTATION_KEY,
    )

    def _action_spec(self) -> ActionSpec:
        return build_action_spec(
            Pos(prefix="right_wrist"),
            Rot("rot6d", prefix="right_wrist"),
            Pos(dim=FINGERTIP_DIM, prefix="right_fingertip"),
            Pos(prefix="left_wrist"),
            Rot("rot6d", prefix="left_wrist"),
            Pos(dim=FINGERTIP_DIM, prefix="left_fingertip"),
        )

    def _build_raw_action(self, rows: list[dict[str, Any]], include_initial_state: bool = False) -> torch.Tensor:
        action = build_webhumanaction_hand_action(
            self._sample_from_rows(rows, self._KEYS),
            pose_convention=self._pose_convention,  # type: ignore[arg-type]
            rotation_format=self._rotation_format,
            include_initial_state=include_initial_state,
        )
        return torch.from_numpy(action).float()


class WebHumanActionBodyLeRobotDataset(_WebHumanActionLeRobotDataset):
    """Action100M body: domain ``webhumanaction_body`` (24), 57D head + hands action."""

    _DOMAIN_NAME = "webhumanaction_body"
    _ACTION_CAPTION_DATASET_NAME = "web_human_action_body"
    _ACTION_DIM = BODY_ACTION_DIM
    _NORMALIZER_PATH = BODY_NORMALIZER_PATH
    _INITIAL_STATE_NORMALIZER_PATH = BODY_INITIAL_STATE_NORMALIZER_PATH
    _KEYS = (
        CAMERA_POSITION_KEY,
        CAMERA_ROTATION_KEY,
        HEAD_POSITION_KEY,
        HEAD_ROTATION_KEY,
        HAND_RIGHT_POSITION_KEY,
        HAND_RIGHT_ROTATION_KEY,
        HAND_LEFT_POSITION_KEY,
        HAND_LEFT_ROTATION_KEY,
    )

    def _action_spec(self) -> ActionSpec:
        return build_action_spec(
            Pos(prefix="head"),
            Rot("rot6d", prefix="head"),
            Pos(prefix="right_wrist"),
            Rot("rot6d", prefix="right_wrist"),
            Pos(dim=FINGERTIP_DIM, prefix="right_fingertip"),
            Pos(prefix="left_wrist"),
            Rot("rot6d", prefix="left_wrist"),
            Pos(dim=FINGERTIP_DIM, prefix="left_fingertip"),
        )

    def _build_raw_action(self, rows: list[dict[str, Any]], include_initial_state: bool = False) -> torch.Tensor:
        action = build_webhumanaction_body_action(
            self._sample_from_rows(rows, self._KEYS),
            pose_convention=self._pose_convention,  # type: ignore[arg-type]
            rotation_format=self._rotation_format,
            include_initial_state=include_initial_state,
        )
        return torch.from_numpy(action).float()


__all__ = [
    "BODY_ACTION_DIM",
    "BODY_INITIAL_STATE_NORMALIZER_PATH",
    "BODY_LAYOUT",
    "BODY_NORMALIZER_PATH",
    "HAND_ACTION_DIM",
    "HAND_INITIAL_STATE_NORMALIZER_PATH",
    "HAND_LAYOUT",
    "HAND_NORMALIZER_PATH",
    "WebHumanActionBodyLeRobotDataset",
    "WebHumanActionHandLeRobotDataset",
    "build_webhumanaction_body_action",
    "build_webhumanaction_hand_action",
]
