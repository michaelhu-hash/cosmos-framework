# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Trajectory exporter: a0 outputs are self-anchored, dataset anchors reproduce the reader's frame-0 poses."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as R

from cosmos_framework.data.generator.action.utils.human_pose_layout import (
    HumanPoseAnchors,
    HumanPoseLayout,
    decode_initial_state_row,
    encode_initial_state_row,
)
from cosmos_framework.data.generator.action.utils.pose_utils import pose_abs_to_rel
from cosmos_framework.scripts import export_human_pose_trajectories as x


def _pose(rng: np.random.Generator) -> np.ndarray:
    p = np.eye(4)
    p[:3, :3] = R.random(random_state=int(rng.integers(1 << 30))).as_matrix()
    p[:3, 3] = rng.normal(scale=0.3, size=3) + [0, 0, 0.6]
    return p


def _a0_window(
    layout: HumanPoseLayout, rng: np.random.Generator, num_steps: int = 16
) -> tuple[np.ndarray, HumanPoseAnchors]:
    """Raw [T+1, D] window: a0 row + static deltas (identity poses, fixed fingertips)."""
    anchors = HumanPoseAnchors(
        wrist_poses=(_pose(rng), _pose(rng)),
        fingers_local=(rng.normal(scale=0.04, size=(5, 3)),) * 2,
        head_pose=_pose(rng) if layout.has_head else None,
    )
    a0 = encode_initial_state_row(layout, anchors)
    static = np.broadcast_to(np.eye(4), (num_steps + 1, 4, 4))
    blocks = []
    if layout.has_camera:
        blocks.append(pose_abs_to_rel(static, rotation_format="rot6d", pose_convention="backward_chunk_anchored_16f"))
    if layout.has_head:
        blocks.append(pose_abs_to_rel(static, rotation_format="rot6d", pose_convention="backward_chunk_anchored_16f"))
    for hand in range(2):
        blocks.append(pose_abs_to_rel(static, rotation_format="rot6d", pose_convention="backward_chunk_anchored_16f"))
        blocks.append(np.broadcast_to(anchors.fingers_local[hand].reshape(-1), (num_steps, 15)))
    rows = np.concatenate(blocks, axis=-1).astype(np.float32)
    return np.concatenate([a0[None], rows], axis=0), anchors


@pytest.mark.L0
@pytest.mark.parametrize("domain", ["webhumanaction_hand", "webhumanaction_body", "hand_pose"])
def test_a0_output_is_self_anchored_and_round_trips_through_the_shipped_stats(domain: str) -> None:
    layout = x._ARMS[domain]["layout"]
    rng = np.random.default_rng(3)
    raw, anchors = _a0_window(layout, rng)
    norm = x.humanaction_normalizer(domain, initial_state=True).normalize_action(torch.as_tensor(raw)).numpy()
    # Padded to max_action_dim like a real CLI output.
    padded = np.concatenate([norm, np.zeros((norm.shape[0], 64 - norm.shape[1]), np.float32)], axis=1)
    out = x.export_trajectories(padded, domain_name=domain, has_initial_state=True, fps=15.0)
    assert out["anchor_source"] == "predicted" and out["num_frames"] == 17 and out["format"] == x.FORMAT
    np.testing.assert_allclose(out["pred_action"], raw, atol=1e-4, rtol=1e-4)
    # Static deltas: every frame equals the a0 anchors.
    for hand in range(2):
        np.testing.assert_allclose(
            out["pred_wrist_poses"][hand], np.broadcast_to(anchors.wrist_poses[hand], (17, 4, 4)), atol=1e-4
        )
        tips = (
            np.einsum("ij,nj->ni", anchors.wrist_poses[hand][:3, :3], anchors.fingers_local[hand])
            + anchors.wrist_poses[hand][:3, 3]
        )
        np.testing.assert_allclose(out["pred_fingertips"][hand], np.broadcast_to(tips, (17, 5, 3)), atol=1e-4)
    if layout.has_head:
        np.testing.assert_allclose(out["pred_head_poses"], np.broadcast_to(anchors.head_pose, (17, 4, 4)), atol=1e-4)
    else:
        assert "pred_head_poses" not in out
    np.testing.assert_allclose(out["pred_camera_poses"], np.broadcast_to(np.eye(4), (17, 4, 4)), atol=1e-6)
    decoded = decode_initial_state_row(out["pred_initial_state_row"], layout)
    np.testing.assert_allclose(decoded.wrist_poses[1], anchors.wrist_poses[1], atol=1e-4)


@pytest.mark.L0
def test_plain_output_without_dataset_falls_back_to_identity_anchors() -> None:
    layout = x._ARMS["webhumanaction_hand"]["layout"]
    raw, _ = _a0_window(layout, np.random.default_rng(5))
    norm = (
        x.humanaction_normalizer("webhumanaction_hand", initial_state=False)
        .normalize_action(torch.as_tensor(raw[1:]))
        .numpy()
    )
    out = x.export_trajectories(norm, domain_name="webhumanaction_hand", has_initial_state=False)
    assert out["anchor_source"] == "identity" and "pred_initial_state_row" not in out
    np.testing.assert_allclose(out["pred_wrist_poses"][:, 0], np.broadcast_to(np.eye(4), (2, 4, 4)), atol=1e-6)
    with pytest.raises(ValueError):
        x.export_trajectories(norm, domain_name="webhumanaction_hand", has_initial_state=False, anchors="predicted")


@pytest.mark.L0
def test_cli_json_loader_and_flag_inference(tmp_path: Path) -> None:
    rows = np.zeros((73, 64), np.float32).tolist()
    f = tmp_path / "sample_outputs.json"
    f.write_text(
        json.dumps(
            {
                "args": {"domain_name": "webhumanaction_hand", "action_chunk_size": 72, "fps": 15},
                "outputs": [{"content": {"action": rows}}],
            }
        )
    )
    action, cli = x.load_cli_output(f)
    assert action.shape == (73, 64) and cli["domain_name"] == "webhumanaction_hand"


@pytest.mark.L1
@pytest.mark.skipif(
    "COSMOS_WEBHUMANACTION_HAND_LEROBOT_ROOT" not in os.environ, reason="needs an Action100M hand LeRobot export"
)
def test_dataset_anchors_export_gt_and_pred_from_a_real_window(tmp_path: Path) -> None:
    root = os.environ["COSMOS_WEBHUMANACTION_HAND_LEROBOT_ROOT"]
    from cosmos_framework.data.generator.action.datasets.webhumanaction_lerobot_dataset import (
        WebHumanActionHandLeRobotDataset,
    )

    reader = WebHumanActionHandLeRobotDataset(root, mode="inverse_dynamics")
    item = reader[0]  # normalized [72, 48], like a perfect CLI prediction
    out = x.export_trajectories(
        item["action"].numpy(),
        domain_name="webhumanaction_hand",
        has_initial_state=False,
        anchors="dataset",
        dataset_root=root,
        window_index=0,
    )
    assert out["anchor_source"] == "dataset" and out["gt_wrist_poses"].shape == (2, 73, 4, 4)
    np.testing.assert_allclose(
        out["pred_wrist_poses"], out["gt_wrist_poses"], atol=1e-4
    )  # perfect prediction == GT chains
    np.testing.assert_allclose(out["pred_fingertips"], out["gt_fingertips"], atol=1e-4)
    npz_path = tmp_path / "traj.npz"
    np.savez_compressed(npz_path, **out)
    loaded = np.load(npz_path, allow_pickle=False)
    assert str(loaded["format"]) == x.FORMAT and list(loaded["hand_order"]) == ["right", "left"]
