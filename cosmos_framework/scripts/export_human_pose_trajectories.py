# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Turn Cosmos3-Nano-HumanAction action-CLI output into camera-frame human-pose trajectories.

The action CLI (``model_mode=inverse_dynamics`` / ``wam``) writes normalized action rows to
``sample_outputs.json``. Downstream motion tools (e.g. the motiongen egorecon handoff) want SE(3)
trajectories instead. This script denormalizes the rows with the arm's shipped statistics, decodes them
with :mod:`cosmos_framework.data.generator.action.utils.human_pose_layout` and writes one ``.npz``::

    format                 "cosmos_human_pose_trajectories_v1"
    domain_name, layout    "webhumanaction_hand" | "webhumanaction_body" | "hand_pose"; HumanPoseLayout preset name
    pose_convention, fps   how the rows were encoded; model fps (15 for the HumanAction recipe)
    frame                  "camera0": every pose is in the frame-0 camera frame (moving cameras keep their
                           residual ego-motion in ``*_camera_poses``; static cameras have identity there)
    anchor_source          "predicted" (Image2Action a0 row) | "dataset" (frame-0 poses of a LeRobot window) | "identity"
    has_initial_state      whether row 0 of ``*_action`` is the a0 row
    hand_order             ["right", "left"]
    <src>_wrist_poses      [2, T+1, 4, 4]   <src> in {pred, gt}; gt only with --dataset-root
    <src>_fingertips       [2, T+1, 5, 3]   thumb, index, middle, ring, pinky (camera frame)
    <src>_head_poses       [T+1, 4, 4]      body layout only
    <src>_camera_poses     [T+1, 4, 4]      camera chain, identity for camera-free layouts
    <src>_action           [T(+1), D]       denormalized rows
    <src>_initial_state_row [D]             the a0 row when present

Anchors: an Image2Action checkpoint (``predict_initial_state=true`` in the CLI spec) predicts the absolute
frame-0 pose itself, so its output is self-contained (``--anchors predicted``, the default for such outputs).
For the plain checkpoints the deltas need frame-0 anchors: give the LeRobot window the clip came from
(``--dataset-root`` + ``--window-index``, ``--anchors dataset``; this also exports the GT trajectories), or accept
``--anchors identity`` (wrists start at the origin; shapes only).

Example::

    python -m cosmos_framework.scripts.export_human_pose_trajectories \\
        --sample-outputs outputs/humanaction_hand_id/0/sample_outputs.json \\
        --dataset-root assets/webhumanaction_hand_lerobot_example --window-index 0 \\
        --output humanaction_hand_trajectories.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cosmos_framework.data.generator.action.datasets.human_hand_pose_lerobot_dataset import (
    HAND_POSE_LAYOUT,
    HUMANACTION_INITIAL_STATE_NORMALIZER_PATH,
    HUMANACTION_NORMALIZER_PATH,
    HumanHandPoseLeRobotDataset,
)
from cosmos_framework.data.generator.action.datasets.webhumanaction_lerobot_dataset import (
    BODY_INITIAL_STATE_NORMALIZER_PATH,
    BODY_LAYOUT,
    BODY_NORMALIZER_PATH,
    HAND_INITIAL_STATE_NORMALIZER_PATH,
    HAND_LAYOUT,
    HAND_NORMALIZER_PATH,
    WebHumanActionBodyLeRobotDataset,
    WebHumanActionHandLeRobotDataset,
)
from cosmos_framework.data.generator.action.utils.action_processing import (
    ActionInitialStateNormalization,
    ActionNormalizer,
    load_action_normalizer,
)
from cosmos_framework.data.generator.action.utils.human_pose_layout import (
    HumanPoseAnchors,
    HumanPoseChains,
    HumanPoseLayout,
    decode_human_pose_chains,
    decode_initial_state_row,
    split_initial_state,
)

FORMAT = "cosmos_human_pose_trajectories_v1"
DEFAULT_POSE_CONVENTION = "backward_chunk_anchored_16f"
DEFAULT_NORMALIZATION = "piecewise_asinh_rot"
HAND_ORDER = ("right", "left")

_ARMS: dict[str, dict[str, Any]] = {
    "webhumanaction_hand": dict(
        layout=HAND_LAYOUT,
        layout_name="hand_pose_camera_free",
        stats=HAND_NORMALIZER_PATH,
        a0_stats=HAND_INITIAL_STATE_NORMALIZER_PATH,
        reader=WebHumanActionHandLeRobotDataset,
    ),
    "webhumanaction_body": dict(
        layout=BODY_LAYOUT,
        layout_name="body_head_wrists",
        stats=BODY_NORMALIZER_PATH,
        a0_stats=BODY_INITIAL_STATE_NORMALIZER_PATH,
        reader=WebHumanActionBodyLeRobotDataset,
    ),
    "hand_pose": dict(
        layout=HAND_POSE_LAYOUT,
        layout_name="hand_pose",
        stats=HUMANACTION_NORMALIZER_PATH,
        a0_stats=HUMANACTION_INITIAL_STATE_NORMALIZER_PATH,
        reader=HumanHandPoseLeRobotDataset,
    ),
}


def humanaction_normalizer(
    domain_name: str,
    *,
    initial_state: bool,
    stats_path: str | Path | None = None,
    initial_state_stats_path: str | Path | None = None,
    method: str = DEFAULT_NORMALIZATION,
) -> ActionNormalizer:
    """The arm's action normalizer (row-aware when the rows carry the a0 row), without a dataset root."""
    arm = _ARMS[domain_name]
    dim = arm["layout"].action_dim
    base = load_action_normalizer(
        method, stats_path=Path(stats_path or arm["stats"]), apply_forward_clamp=False, expected_dim=dim
    )  # type: ignore[arg-type]
    if not initial_state:
        return base
    a0 = load_action_normalizer(
        method,
        stats_path=Path(initial_state_stats_path or arm["a0_stats"]),
        apply_forward_clamp=False,
        expected_dim=dim,
    )  # type: ignore[arg-type]
    return ActionInitialStateNormalization(base=base, initial_state=a0)


def identity_anchors(layout: HumanPoseLayout) -> HumanPoseAnchors:
    eye = np.eye(4)
    zeros = np.zeros((layout.num_finger_joints, 3))
    return HumanPoseAnchors(
        wrist_poses=(eye.copy(), eye.copy()),
        fingers_local=(zeros.copy(), zeros.copy()),
        head_pose=eye.copy() if layout.has_head else None,
    )


def chains_to_arrays(
    prefix: str, chains: HumanPoseChains, layout: HumanPoseLayout, num_frames: int
) -> dict[str, np.ndarray]:
    out = {
        f"{prefix}_wrist_poses": np.stack(chains.wrist_poses).astype(np.float64),  # [2,T+1,4,4] right, left
        f"{prefix}_fingertips": np.stack(chains.fingertips).astype(np.float64),  # [2,T+1,J,3]
        f"{prefix}_camera_poses": (
            chains.camera_poses if chains.camera_poses is not None else np.broadcast_to(np.eye(4), (num_frames, 4, 4))
        ).astype(np.float64),
    }
    if layout.has_head:
        assert chains.head_poses is not None
        out[f"{prefix}_head_poses"] = chains.head_poses.astype(np.float64)
    return out


def export_trajectories(
    action_norm: np.ndarray,
    *,
    domain_name: str,
    has_initial_state: bool,
    pose_convention: str = DEFAULT_POSE_CONVENTION,
    fps: float = 15.0,
    anchors: str = "auto",
    dataset_root: str | Path | None = None,
    window_index: int = 0,
    stats_path: str | Path | None = None,
    initial_state_stats_path: str | Path | None = None,
) -> dict[str, Any]:
    """Core of the script: normalized CLI rows -> the trajectory npz payload (a dict of arrays / scalars)."""
    if domain_name not in _ARMS:
        raise ValueError(f"domain_name must be one of {sorted(_ARMS)}, got {domain_name!r}")
    arm = _ARMS[domain_name]
    layout: HumanPoseLayout = arm["layout"]
    action_norm = np.asarray(action_norm, dtype=np.float32)
    if action_norm.ndim != 2:
        raise ValueError(f"expected [T(+1), D] action rows, got shape {action_norm.shape}")
    action_norm = action_norm[:, : layout.action_dim]  # drop max_action_dim padding
    normalizer = humanaction_normalizer(
        domain_name,
        initial_state=has_initial_state,
        stats_path=stats_path,
        initial_state_stats_path=initial_state_stats_path,
    )
    pred_raw = normalizer.denormalize_action(torch.as_tensor(action_norm)).numpy().astype(np.float64)  # [T(+1),D]

    if anchors == "auto":
        anchors = "predicted" if has_initial_state else ("dataset" if dataset_root is not None else "identity")
    payload: dict[str, Any] = {
        "format": FORMAT,
        "domain_name": domain_name,
        "layout": arm["layout_name"],
        "pose_convention": pose_convention,
        "fps": float(fps),
        "frame": "camera0",
        "anchor_source": anchors,
        "has_initial_state": bool(has_initial_state),
        "hand_order": np.array(HAND_ORDER),
        "pred_action": pred_raw.astype(np.float32),
    }

    gt_anchors: HumanPoseAnchors | None = None
    if anchors == "dataset" or dataset_root is not None:
        if dataset_root is None:
            raise ValueError("--anchors dataset needs --dataset-root")
        num_steps = pred_raw.shape[0] - int(has_initial_state)
        reader_kwargs: dict[str, Any] = dict(
            chunk_length=num_steps,
            pose_convention=pose_convention,
            mode="inverse_dynamics",
            action_normalization=DEFAULT_NORMALIZATION,
            fps=fps,
        )
        if domain_name == "hand_pose":
            reader_kwargs["stats_path"] = Path(stats_path or arm["stats"])
        reader = arm["reader"](str(dataset_root), **reader_kwargs)
        gt_with_a0 = reader.raw_action_window(window_index, include_initial_state=True).astype(np.float64)  # [T+1,D]
        gt_a0, gt_rows = split_initial_state(gt_with_a0)
        gt_anchors = decode_initial_state_row(gt_a0, layout)
        gt_chains = decode_human_pose_chains(gt_rows, layout, gt_anchors, pose_convention=pose_convention)
        payload.update(chains_to_arrays("gt", gt_chains, layout, gt_rows.shape[0] + 1))
        payload["gt_action"] = gt_rows.astype(np.float32)
        payload["gt_initial_state_row"] = gt_a0.astype(np.float32)
        payload["dataset_root"] = str(dataset_root)
        payload["window_index"] = int(window_index)

    if has_initial_state:
        pred_a0, pred_rows = split_initial_state(pred_raw)
        payload["pred_initial_state_row"] = pred_a0.astype(np.float32)
    else:
        pred_a0, pred_rows = None, pred_raw
    if anchors == "predicted":
        if pred_a0 is None:
            raise ValueError("--anchors predicted needs an Image2Action output (predict_initial_state=true, T+1 rows)")
        pred_anchors = decode_initial_state_row(pred_a0, layout)
    elif anchors == "dataset":
        assert gt_anchors is not None
        pred_anchors = gt_anchors
    elif anchors == "identity":
        pred_anchors = identity_anchors(layout)
    else:
        raise ValueError(f"unknown anchors mode {anchors!r}")
    pred_chains = decode_human_pose_chains(pred_rows, layout, pred_anchors, pose_convention=pose_convention)
    payload.update(chains_to_arrays("pred", pred_chains, layout, pred_rows.shape[0] + 1))
    payload["num_frames"] = int(pred_rows.shape[0] + 1)
    return payload


def load_cli_output(path: str | Path, output_index: int = 0) -> tuple[np.ndarray, dict[str, Any]]:
    """``(normalized action rows [T(+1), D], CLI args)`` from a ``sample_outputs.json``."""
    data = json.loads(Path(path).read_text())
    action = np.asarray(data["outputs"][output_index]["content"]["action"], dtype=np.float32)
    return action, dict(data.get("args") or {})


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-outputs", required=True, help="sample_outputs.json written by the action CLI (ID / wam)")
    ap.add_argument("--output-index", type=int, default=0, help="which entry of outputs[] to export")
    ap.add_argument("--output", required=True, help="output .npz")
    ap.add_argument("--domain", default=None, choices=sorted(_ARMS), help="default: the CLI args' domain_name")
    ap.add_argument(
        "--predict-initial-state",
        choices=("auto", "true", "false"),
        default="auto",
        help="whether row 0 is the a0 row; auto reads the CLI args, else infers from the row count",
    )
    ap.add_argument("--pose-convention", default=DEFAULT_POSE_CONVENTION)
    ap.add_argument("--fps", type=float, default=None, help="default: the CLI args' fps, else 15")
    ap.add_argument("--anchors", choices=("auto", "predicted", "dataset", "identity"), default="auto")
    ap.add_argument("--dataset-root", default=None, help="LeRobot root of the clip (GT anchors + GT trajectories)")
    ap.add_argument("--window-index", type=int, default=0)
    ap.add_argument("--stats-path", default=None)
    ap.add_argument("--initial-state-stats-path", default=None)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    action, cli = load_cli_output(args.sample_outputs, args.output_index)
    domain = args.domain or str(cli.get("domain_name") or "")
    if not domain:
        raise SystemExit("--domain is required (the CLI args carry no domain_name)")
    chunk = cli.get("action_chunk_size")
    if args.predict_initial_state == "auto":
        if "predict_initial_state" in cli:
            has_a0 = bool(cli["predict_initial_state"])
        elif chunk is not None:
            has_a0 = action.shape[0] == int(chunk) + 1
        else:
            has_a0 = False
    else:
        has_a0 = args.predict_initial_state == "true"
    fps = args.fps if args.fps is not None else float(cli.get("fps") or 15.0)
    payload = export_trajectories(
        action,
        domain_name=domain,
        has_initial_state=has_a0,
        pose_convention=args.pose_convention,
        fps=fps,
        anchors=args.anchors,
        dataset_root=args.dataset_root,
        window_index=args.window_index,
        stats_path=args.stats_path,
        initial_state_stats_path=args.initial_state_stats_path,
    )
    payload["sample_outputs"] = str(Path(args.sample_outputs).resolve())
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    right = payload["pred_wrist_poses"][0, :, :3, 3]
    print(
        f"wrote {out}: {payload['num_frames']} frames @ {fps:g} fps, {domain} / {payload['layout']}, anchors={payload['anchor_source']}, "
        f"right wrist z range {right[:, 2].min():.2f}..{right[:, 2].max():.2f} m"
    )


if __name__ == "__main__":
    main()
