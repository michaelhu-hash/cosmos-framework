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
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow.parquet as pq
import torch
from lerobot.datasets.video_utils import decode_video_frames

from cosmos_framework.data.generator.action.datasets.base_dataset import ActionBaseDataset
from cosmos_framework.data.generator.action.utils.action_spec import ActionSpec, Pos, Rot, build_action_spec
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


def fingertips_in_wrist_frame(positions: np.ndarray, wrist_poses: np.ndarray) -> np.ndarray:
    """Five fingertip positions expressed in the aligned wrist frame of the same frame, ``[T, 15]``.

    Uses frames ``1..T`` so the block lines up with the relative-pose rows, which spend frame 0 as the first anchor.
    """
    future = positions[1:].reshape(-1, NUM_JOINTS, 3)  # [T,21,3]
    tips = future[:, FINGERTIP_JOINT_IDXS, :]  # [T,5,3]
    tips_h = np.concatenate([tips, np.ones((*tips.shape[:-1], 1), dtype=np.float32)], axis=-1)  # [T,5,4]
    wrist_inv = np.linalg.inv(wrist_poses[1:])  # [T,4,4]
    tips_wrist = np.einsum("tij,tnj->tni", wrist_inv, tips_h)[..., :3]  # [T,5,3]
    return tips_wrist.reshape(len(future), -1).astype(np.float32)  # [T,15]


def _hand_blocks(
    sample: dict[str, Any],
    *,
    pose_convention: PoseConvention,
    rotation_format: RotationConvention,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shared Action100M hand chain: ``(camera_c2w [T+1,4,4], right_hand [T,24], left_hand [T,24])``."""
    camera_c2w = build_abs_pose_from_components(
        _as_float32(sample[CAMERA_POSITION_KEY], 3, CAMERA_POSITION_KEY),
        _as_float32(sample[CAMERA_ROTATION_KEY], 4, CAMERA_ROTATION_KEY),
        "quat_xyzw",
    )  # [T+1,4,4]
    hands = []
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
    return camera_c2w, hands[0], hands[1]


def build_webhumanaction_hand_action(
    sample: dict[str, Any],
    *,
    pose_convention: PoseConvention = DEFAULT_POSE_CONVENTION,
    rotation_format: RotationConvention = DEFAULT_ROTATION_FORMAT,
) -> np.ndarray:
    """48D hand action ``[right_wrist, right_fingertips, left_wrist, left_fingertips]`` from ``T+1`` frames."""
    _, right_hand, left_hand = _hand_blocks(sample, pose_convention=pose_convention, rotation_format=rotation_format)
    return np.concatenate([right_hand, left_hand], axis=-1).astype(np.float32)  # [T,48]


def build_webhumanaction_body_action(
    sample: dict[str, Any],
    *,
    pose_convention: PoseConvention = DEFAULT_POSE_CONVENTION,
    rotation_format: RotationConvention = DEFAULT_ROTATION_FORMAT,
) -> np.ndarray:
    """57D body action ``[head, right_wrist, right_fingertips, left_wrist, left_fingertips]`` from ``T+1`` frames."""
    camera_c2w, right_hand, left_hand = _hand_blocks(
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
    return np.concatenate([head_rel, right_hand, left_hand], axis=-1).astype(np.float32)  # [T,57]


class _WebHumanActionLeRobotDataset(ActionBaseDataset):
    """Window sampling, video decoding and captions shared by the hand and body arms."""

    _DOMAIN_NAME: str
    _ACTION_DIM: int
    _NORMALIZER_PATH: Path

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
        )
        self._rotation_format: RotationConvention = rotation_format
        source_fps = float(self._info["fps"])
        source_stride = source_fps / self._fps
        if not source_stride.is_integer():
            raise ValueError(f"Source FPS {source_fps} must be an integer multiple of target FPS {self._fps}.")
        self._source_stride = int(source_stride)
        self._image_key = image_key
        required_source_steps = self._source_stride * self._chunk_length
        self._valid_starts: list[int] = []
        episode_start = 0
        while episode_start < len(self._rows):
            episode_index = int(self._rows[episode_start]["episode_index"])
            episode_end = episode_start + 1
            while episode_end < len(self._rows) and int(self._rows[episode_end]["episode_index"]) == episode_index:
                episode_end += 1
            self._valid_starts.extend(
                range(episode_start, max(episode_start, episode_end - required_source_steps), self._sample_stride)
            )
            episode_start = episode_end
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

    def __len__(self) -> int:
        return len(self._valid_starts)

    def _window_rows(self, idx: int) -> list[dict[str, Any]]:
        start = self._valid_starts[int(idx)]
        stop = start + self._source_stride * self._chunk_length + 1
        rows = self._rows[start : stop : self._source_stride]
        if len(rows) != self._chunk_length + 1:
            raise IndexError(f"Incomplete WebHumanAction window at index {idx}.")
        episode_index = int(rows[0]["episode_index"])
        if any(int(row["episode_index"]) != episode_index for row in rows):
            raise IndexError(f"WebHumanAction window at index {idx} crosses an episode boundary.")
        return rows

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

    def _build_raw_action(self, rows: list[dict[str, Any]]) -> torch.Tensor:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode = self._choose_mode()
        rows = self._window_rows(idx)
        episode = self._episodes[int(rows[0]["episode_index"])]
        video = self._load_video(episode, rows)
        raw_action = self._build_raw_action(rows)
        if raw_action.shape != (self._chunk_length, self.action_dim):
            raise ValueError(
                f"Expected action shape {(self._chunk_length, self.action_dim)}, got {tuple(raw_action.shape)}."
            )
        return self._build_result(mode=mode, video=video, action=raw_action, ai_caption=self._caption(rows))


class WebHumanActionHandLeRobotDataset(_WebHumanActionLeRobotDataset):
    """Action100M hands: domain ``webhumanaction_hand`` (31), 48D camera-free action."""

    _DOMAIN_NAME = "webhumanaction_hand"
    _ACTION_DIM = HAND_ACTION_DIM
    _NORMALIZER_PATH = HAND_NORMALIZER_PATH
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

    def _build_raw_action(self, rows: list[dict[str, Any]]) -> torch.Tensor:
        action = build_webhumanaction_hand_action(
            self._sample_from_rows(rows, self._KEYS),
            pose_convention=self._pose_convention,  # type: ignore[arg-type]
            rotation_format=self._rotation_format,
        )
        return torch.from_numpy(action).float()


class WebHumanActionBodyLeRobotDataset(_WebHumanActionLeRobotDataset):
    """Action100M body: domain ``webhumanaction_body`` (24), 57D head + hands action."""

    _DOMAIN_NAME = "webhumanaction_body"
    _ACTION_DIM = BODY_ACTION_DIM
    _NORMALIZER_PATH = BODY_NORMALIZER_PATH
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

    def _build_raw_action(self, rows: list[dict[str, Any]]) -> torch.Tensor:
        action = build_webhumanaction_body_action(
            self._sample_from_rows(rows, self._KEYS),
            pose_convention=self._pose_convention,  # type: ignore[arg-type]
            rotation_format=self._rotation_format,
        )
        return torch.from_numpy(action).float()


__all__ = [
    "BODY_ACTION_DIM",
    "BODY_NORMALIZER_PATH",
    "HAND_ACTION_DIM",
    "HAND_NORMALIZER_PATH",
    "WebHumanActionBodyLeRobotDataset",
    "WebHumanActionHandLeRobotDataset",
    "build_webhumanaction_body_action",
    "build_webhumanaction_hand_action",
]
