#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# ============================================================================
# Structured-TOML launch for Cosmos3-Nano-HumanAction joint post-training.
# Drives cosmos_framework.scripts.train against
# examples/toml/sft_config/action_humanaction_joint_nano.toml (selects the registered
# `action_humanaction_joint_nano` experiment: Action100M hands 48D + Action100M body 57D +
# Mecka hands 57D, 15 fps, 72-step chunks, joint FD / ID / wam sampling).
# See docs/action_humanaction_posttrain.md.
#
# Env vars (override for your filesystem):
#   HUMANACTION_HAND_ROOT   Action100M hands LeRobot root (observation.state.hand_{left,right}_cam*, camera_*)
#   HUMANACTION_BODY_ROOT   Action100M body LeRobot root (+ observation.state.head_cam*)
#   HUMANACTION_MECKA_ROOT  Mecka hands LeRobot root (hand_pose layout)
#   BASE_CHECKPOINT_PATH    DCP dir of nvidia/Cosmos3-Nano or nvidia/Cosmos3-Nano-HumanAction (convert_model_to_dcp)
#   WAN_VAE_PATH            Wan2.2 VAE .pth (Wan-AI/Wan2.2-TI2V-5B)
#   WANDB_API_KEY           for online logging (TOML wandb_mode="online")
#   NPROC_PER_NODE          torchrun --nproc_per_node (default 8)
#   EXTRA_TAIL_OVERRIDES    space-separated Hydra overrides
#
# Single-node smoke (config/data sanity, a few iters):
#   export EXTRA_TAIL_OVERRIDES="trainer.max_iter=10 checkpoint.save_iter=10"
#   bash examples/launch_sft_action_humanaction_joint_nano.sh
#
# The released recipe ran 16 ranks (4 x 4 GPUs) at one packed 74k-token batch per rank
# (~9 windows per rank per step); scale trainer.max_iter with your rank count.
# ============================================================================
TOML_FILE="examples/toml/sft_config/action_humanaction_joint_nano.toml"
: "${HUMANACTION_HAND_ROOT:=examples/data/webhumanaction_hand_lerobot}"
: "${HUMANACTION_BODY_ROOT:=examples/data/webhumanaction_body_lerobot}"
: "${HUMANACTION_MECKA_ROOT:=examples/data/human_hand_pose_lerobot}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Nano}"
export HUMANACTION_HAND_ROOT HUMANACTION_BODY_ROOT HUMANACTION_MECKA_ROOT
EXTRA_DATASET_CHECK='for r in "$HUMANACTION_HAND_ROOT" "$HUMANACTION_BODY_ROOT" "$HUMANACTION_MECKA_ROOT"; do [[ -f "$r/meta/info.json" ]] || { echo "ERROR: missing $r/meta/info.json (a LeRobot v3 root; see docs/action_humanaction_posttrain.md)" >&2; exit 1; }; done'
TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)
source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
