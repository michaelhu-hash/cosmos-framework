# Cosmos3-Nano-HumanAction Post-Training

This document describes how to post-train [Cosmos3-Nano-HumanAction](https://huggingface.co/nvidia/Cosmos3-Nano-HumanAction)
(or start from [Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano)) on human-motion data in `cosmos_framework`,
with the released joint recipe: Action100M hands, Action100M body and Mecka hands trained together, and the
Image2Action variant that also predicts the frame-0 pose (`a0`). Experiments `action_humanaction_joint_nano` and
`action_humanaction_joint_a0_nano`, launchers `launch_sft_action_humanaction_joint_nano.sh` and
`launch_sft_action_humanaction_joint_a0_nano.sh`.

## Overview

| Piece              | Value |
| ------------------ | ----- |
| arms / domains     | `webhumanaction_hand` (31, 48D), `webhumanaction_body` (24, 57D head + hands), `hand_pose` Mecka (3, 57D camera + hands) |
| data               | LeRobot v3 roots, one per arm (see [Data Layout](#data-layout)); the 21-joint xyzw hand pose exports of Action100M / Mecka |
| action encoding    | 15 fps, 72-step chunks (73 frames), `backward_chunk_anchored_16f` rot6d deltas, fingertips in the aligned wrist frame |
| normalization      | `piecewise_asinh_rot` with the shipped per-arm stats under `cosmos_framework/data/generator/action/normalizer_stats/` |
| task mix           | `mode="joint"`: forward_dynamics / inverse_dynamics / wam drawn per sample; Mecka samples one window per subtask |
| a0 variant         | `initial_state="predict"`: inverse dynamics / wam windows carry the absolute frame-0 pose as row 0 (own stats, loss weight 10) |
| init               | `Cosmos3-Nano` (GA mid-training) or `Cosmos3-Nano-HumanAction`, converted to DCP |
| resolution / batch | 480p, token-budget packing (74k tokens per rank ≈ 9 windows); released run: 16 ranks, 25k iterations, lr 1e-4 |

## Prerequisites

- Install the training environment as described in [`docs/setup.md`](./setup.md).
- Run commands from the repository root.

## Inputs You Provide

This package ships the training stack: the two registered experiments, the three dataset classes
(`WebHumanActionHandLeRobotDataset`, `WebHumanActionBodyLeRobotDataset`, `HumanHandPoseLeRobotDataset`), the shipped
normalizer statistics and the paired TOML / launch shells. You provide:

1. **Three LeRobot v3 roots** (`HUMANACTION_HAND_ROOT`, `HUMANACTION_BODY_ROOT`, `HUMANACTION_MECKA_ROOT`). To train a
   single arm, set the other ratios to 0, e.g.
   `EXTRA_TAIL_OVERRIDES="dataloader_train.dataloader.datasets.hand_pose_mecka.ratio=0 dataloader_train.dataloader.datasets.webhumanaction_body.ratio=0"`
   (the roots must still exist).
2. **A DCP base checkpoint** (`BASE_CHECKPOINT_PATH`): convert `nvidia/Cosmos3-Nano` (fresh post-training) or
   `nvidia/Cosmos3-Nano-HumanAction` (continue from the release) with `convert_model_to_dcp` (see below).
3. **The Wan2.2 VAE** (`WAN_VAE_PATH`, `Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth`).
4. **Normalizer statistics for your data** (optional). The shipped stats describe Action100M / Mecka; for data with a
   different pose distribution compute your own and point the recipe at them (see [Normalizer statistics](#normalizer-statistics)).

## Data Layout

Each root is a LeRobot v3 dataset (`meta/info.json`, `meta/episodes/`, `meta/tasks.parquet`, optional
`meta/subtasks.parquet`, `data/chunk-*/file-*.parquet`, `videos/`). Required per-frame state features (float arrays):

| Arm                  | Features |
| -------------------- | -------- |
| hands / Mecka        | `observation.state.hand_{right,left}_cam` (21 joints × xyz, wrist first), `observation.state.hand_{right,left}_cam_rotation` (21 × xyzw), `observation.state.camera_position` (3), `observation.state.camera_rotation` (xyzw) |
| body                 | the above + `observation.state.head_cam` (3) and `observation.state.head_cam_rotation` (xyzw) |
| video                | `observation.images.main`; source fps must be an integer multiple of 15 |
| captions             | `task_index` → `meta/tasks.parquet`; per-frame `subtask_index` → `meta/subtasks.parquet` when present (captions are drawn from the subtask text, ` \| ` separates alternatives) |

The example assets under the cookbook (`webhumanaction_{hand,body}_lerobot_example`, `human_hand_pose_lerobot_example`)
show the exact schema. Quality filtering (out-of-frame hands, tracking quality) is done when exporting the data, not by
the readers.

## Normalizer statistics

```shell
python -m cosmos_framework.scripts.compute_action_normalizer_stats \
  --dataset webhumanaction_hand --root $HUMANACTION_HAND_ROOT \
  --output my_hand_stats.json --initial-state-output my_hand_initial_state_stats.json
# --dataset webhumanaction_body / human_hand_pose (add --snap-to-subtask for Mecka) likewise.
```

Then override the paths per dataset, e.g.
`EXTRA_TAIL_OVERRIDES="dataloader_train.dataloader.datasets.webhumanaction_hand.dataset.stats_path=my_hand_stats.json dataloader_train.dataloader.datasets.webhumanaction_hand.dataset.initial_state_stats_path=my_hand_initial_state_stats.json"`.
The tool reads only the action columns (no video decoding) and writes the same JSON format as the shipped files.

## Full Reproduction

```shell
# Step 1: point at the three LeRobot roots.
export HUMANACTION_HAND_ROOT=/path/to/webhumanaction_hand_lerobot
export HUMANACTION_BODY_ROOT=/path/to/webhumanaction_body_lerobot
export HUMANACTION_MECKA_ROOT=/path/to/human_hand_pose_lerobot

# Step 2: convert the base checkpoint -> $BASE_CHECKPOINT_PATH (Cosmos3-Nano, or Cosmos3-Nano-HumanAction to continue).
export BASE_CHECKPOINT_PATH=examples/checkpoints/Cosmos3-Nano-HumanAction-dcp
python -m cosmos_framework.scripts.convert_model_to_dcp \
  --checkpoint-path Cosmos3-Nano-HumanAction \
  -o $BASE_CHECKPOINT_PATH
export WAN_VAE_PATH=/path/to/Wan2.2_VAE.pth

# Step 3: choose the output root and launch (8 GPUs by default; NPROC_PER_NODE=4 on 4-GPU hosts, then also
# EXTRA_TAIL_OVERRIDES="model.config.parallelism.data_parallel_shard_degree=4").
export IMAGINAIRE_OUTPUT_ROOT=/path/to/output_root
bash examples/launch_sft_action_humanaction_joint_nano.sh      # released recipe
bash examples/launch_sft_action_humanaction_joint_a0_nano.sh   # Image2Action (a0) variant
```

The TOML (`examples/toml/sft_config/action_humanaction_joint_nano.toml`) carries the scalar knobs (`max_iter`,
`save_iter`, parallelism, wandb); the dataset / action knobs come from the registered experiment. The released run used
16 ranks (4 nodes × 4 GPUs) for 25k iterations; with fewer ranks scale `trainer.max_iter` up accordingly.

## Validate The Config

```shell
python -m cosmos_framework.scripts.train \
  --sft-toml examples/toml/sft_config/action_humanaction_joint_nano.toml --dryrun
```

## Outputs

Training outputs land under `$IMAGINAIRE_OUTPUT_ROOT/cosmos3/action_sft/<experiment>/checkpoints/iter_*/` (DCP).
Export with `cosmos_framework.scripts.export_model` / `convert_model_to_diffusers` for the inference CLIs.

## Notes

- **a0 inference.** Checkpoints trained with `action_humanaction_joint_a0_nano` must be run with
  `predict_initial_state=true` in the action spec: inverse dynamics / wam then return `action_chunk_size + 1` rows,
  row 0 being the absolute camera-frame pose at frame 0. Denormalize with the reader (`denormalize`, row-aware), then
  `human_pose_layout.split_initial_state` → `decode_initial_state_row` → `decode_human_pose_chains` gives camera-frame
  head / wrist / fingertip trajectories, the input for downstream motion generation. Forward dynamics is unchanged.
- **Splits.** The readers hold out an episode fraction (`val_ratio`, seeded) so `split="val"` can be evaluated with the
  same class; the released fractions are tiny because the internal corpora are large. Tiny datasets round to zero
  held-out episodes.
- **Mecka windows.** `snap_to_subtask=True` samples one window per subtask, starting at the subtask boundary, with a
  variable length (up to 73 frames, 1 + 4N). The Action100M arms use dense sliding windows.
- **Mixing.** `ratio` per dataset weights the three streams (1 : 1 : 1 in the released run).
- **Small datasets / smoke runs.** The streaming loader shards whole episodes across `ranks × num_workers`; with
  fewer episodes than shards it falls back to sharding individual windows, so a one-episode example still feeds
  every rank. Use `dataloader_train.dataloader.num_workers=4` (or lower) on small hosts.
