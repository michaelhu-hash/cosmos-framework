# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""``action_humanaction_joint_nano`` / ``action_humanaction_joint_a0_nano`` — Cosmos3-Nano-HumanAction post-training.

The released HumanAction recipe: three human-motion arms trained jointly from the Cosmos3-Nano base
(GA mid-training checkpoint) — Action100M hands (``webhumanaction_hand``, 48D), Action100M body
(``webhumanaction_body``, 57D head + hands) and Mecka hands (``hand_pose``, 57D camera + hands) — at 15 fps,
72-step chunks (73 frames) at 480p, ``backward_chunk_anchored_16f`` rot6d deltas, ``piecewise_asinh_rot``
normalization with the shipped per-arm stats, JSON prompts, and ``mode="joint"`` (forward_dynamics /
inverse_dynamics / wam drawn per sample). Mecka samples one window per subtask (``snap_to_subtask``).

``action_humanaction_joint_a0_nano`` is the Image2Action variant: every arm also carries the frame-0
initial-state row a0 (``initial_state="predict"``, dedicated a0 stats, a0 loss weight 10), so inverse
dynamics / wam predict the absolute frame-0 pose instead of being given it.

Usage (1 node, 8 GPU)::

    HUMANACTION_HAND_ROOT=... HUMANACTION_BODY_ROOT=... HUMANACTION_MECKA_ROOT=... \\
    BASE_CHECKPOINT_PATH=<Cosmos3-Nano or Cosmos3-Nano-HumanAction DCP dir> WAN_VAE_PATH=<Wan2.2_VAE.pth> \\
    bash examples/launch_sft_action_humanaction_joint_nano.sh

The released run used 16 ranks (4 x 4 GPUs) with one packed 74k-token batch per rank (~9 windows per rank
per step, i.e. ~150 windows per optimizer step) for 25k iterations; scale ``trainer.max_iter`` accordingly.
"""

import copy
from typing import Any

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import (
    get_action_human_hand_pose_sft_dataset,
    get_action_webhumanaction_body_sft_dataset,
    get_action_webhumanaction_hand_sft_dataset,
)
from cosmos_framework.data.generator.joint_dataloader import (
    PackingDataLoader,
    RankPartitionedDataLoader,
)
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

cs = ConfigStore.instance()

# Per-arm held-out episode fractions of the released run (episode-level, seeded).
_VAL_RATIO = {"hand": 0.01, "body": 0.002, "mecka": 0.00012}
_TRAINING_ITERATIONS = 25_000


def _humanaction_datasets(*, initial_state: str | None) -> dict[str, Any]:
    common = dict(
        fps=15.0,
        chunk_length=72,
        mode="joint",
        pose_convention="backward_chunk_anchored_16f",
        action_normalization="piecewise_asinh_rot",
        initial_state=initial_state,
        split="train",
        split_seed=42,
        resolution="480",
        max_action_dim="${model.config.max_action_dim}",
        tokenizer_config="${model.config.vlm_config.tokenizer}",
        cfg_dropout_rate=0.1,
        format_prompt_as_json=True,
        iterable_shuffle=True,
        episode_shuffle_seed=42,
    )
    return dict(
        webhumanaction_hand=dict(
            ratio=1.0,
            dataset=L(get_action_webhumanaction_hand_sft_dataset)(
                root="${oc.env:HUMANACTION_HAND_ROOT}", val_ratio=_VAL_RATIO["hand"], snap_to_subtask=False, **common
            ),
        ),
        hand_pose_mecka=dict(
            ratio=1.0,
            dataset=L(get_action_human_hand_pose_sft_dataset)(
                root="${oc.env:HUMANACTION_MECKA_ROOT}", val_ratio=_VAL_RATIO["mecka"], snap_to_subtask=True, **common
            ),
        ),
        webhumanaction_body=dict(
            ratio=1.0,
            dataset=L(get_action_webhumanaction_body_sft_dataset)(
                root="${oc.env:HUMANACTION_BODY_ROOT}", val_ratio=_VAL_RATIO["body"], snap_to_subtask=False, **common
            ),
        ),
    )


def _make_experiment(name: str, *, initial_state: str | None) -> LazyDict:
    return LazyDict(
        dict(
            defaults=[
                {"override /model": "mot_fsdp"},
                {"override /data_train": None},
                {"override /data_val": None},
                {"override /optimizer": "fusedadamw"},
                {"override /scheduler": "lambdalinear"},
                {"override /checkpoint": "s3"},
                {"override /callbacks": ["basic", "optimization", "job_monitor"]},
                {"override /ema": "power"},
                {"override /tokenizer": "wan2pt2_tokenizer"},
                {"override /sound_tokenizer": None},
                {"override /vlm_config": None},
                {"override /ckpt_type": "dcp"},
                "_self_",
            ],
            job=dict(project="cosmos3", group="action_sft", name=name, wandb_mode="disabled"),
            model=dict(config=copy.deepcopy(NANO_MODEL_CONFIG)),  # action_gen=True, max_action_dim=64
            optimizer=dict(
                betas=[0.9, 0.99],
                eps=1.0e-08,
                fused=True,
                keys_to_select=[
                    "moe_gen",
                    "time_embedder",
                    "vae2llm",
                    "llm2vae",
                    "action2llm",
                    "llm2action",
                    "action_modality_embed",
                ],
                lr=1.0e-04,
                # Released run: action heads at 5x the base LR.
                lr_multipliers={"action2llm": 5.0, "llm2action": 5.0, "action_modality_embed": 5.0},
                optimizer_type="FusedAdam",
                weight_decay=0.05,
            ),
            scheduler=dict(
                lr_scheduler_type="LambdaLinear",
                cycle_lengths=[_TRAINING_ITERATIONS],
                f_max=[0.4],
                f_min=[0.0],
                f_start=[0.0],
                verbosity_interval=0,
                warm_up_steps=[0],
            ),
            trainer=dict(
                distributed_parallelism="fsdp",
                grad_accum_iter=1,
                logging_iter=50,
                max_iter=_TRAINING_ITERATIONS,
                max_val_iter=None,
                run_validation=False,
                run_validation_on_start=False,
                save_zero_checkpoint=False,
                seed=42,
                timeout_period=999999999,
                validation_iter=100,
                compile_config=dict(recompile_limit=8, use_duck_shape=False),
                cudnn=dict(benchmark=True, deterministic=False),
                ddp=dict(broadcast_buffers=True, find_unused_parameters=False, static_graph=True),
                grad_scaler_args=dict(enabled=False),
                callbacks=dict(
                    dataloader_speed=dict(every_n=100, save_s3=False, step_size=1),
                    device_monitor=dict(
                        every_n=200, log_memory_detail=True, save_s3=False, step_size=1, upload_every_n_mul=5
                    ),
                    grad_clip=dict(clip_norm=1.0, force_finite=True),
                    heart_beat=dict(every_n=200, save_s3=False, step_size=1, update_interval_in_minute=20),
                    iter_speed=dict(every_n=1, hit_thres=50, save_s3=False, save_s3_every_log_n=500),
                    low_precision=dict(update_iter=1),
                    manual_gc=dict(every_n=5, gc_level=1, warm_up=1),
                    param_count=dict(save_s3=False),
                    skip_nan_step=dict(max_consecutive_nan=100),
                    training_stats=dict(log_freq=100),
                ),
            ),
            checkpoint=dict(
                broadcast_via_filesystem=False,
                dcp_async_mode_enabled=False,
                enable_gcs_patch_in_boto3=True,
                keys_not_to_resume=[],
                # Warm start from Cosmos3-Nano (GA mid-training) or Cosmos3-Nano-HumanAction: the action heads are
                # trained in both, so only the EMA shadow is skipped.
                keys_to_skip_loading=["net_ema."],
                load_ema_to_reg=False,
                load_path="???",  # DCP dir; supply via TOML/env
                load_training_state=False,
                only_load_scheduler_state=False,
                save_iter=250,
                strict_resume=False,
                verbose=True,
                hf_export=dict(
                    enabled=False,
                    export_every_n=1,
                    hf_repo_id=None,
                    upload_to_object_store=dict(bucket="", credentials="", enabled=False),
                ),
                jit=dict(device="cuda", dtype="bfloat16", enabled=False, input_shape=None, strict=True),
                load_from_object_store=dict(bucket="", credentials="", enabled=False),
                save_to_object_store=dict(bucket="", credentials="", enabled=False),
            ),
            dataloader_train=L(PackingDataLoader)(
                audio_sample_rate=48000,
                dataset_name="action_humanaction",
                # Released run: token-budget packing (one ~74k-token pack per rank per step), not a sample count.
                max_samples_per_batch=None,
                max_sequence_length="${model.config.max_num_tokens_after_packing}",
                patch_spatial=2,
                sound_latent_fps=0,
                tokenizer_spatial_compression_factor=16,
                tokenizer_temporal_compression_factor=4,
                dataloader=L(RankPartitionedDataLoader)(
                    batch_size=1,
                    in_order=False,
                    num_workers=16,  # host-CPU-bound (video decode); lower via CLI on smaller hosts
                    persistent_workers=True,
                    pin_memory=True,
                    prefetch_factor=2,
                    sampler=None,
                    datasets=_humanaction_datasets(initial_state=initial_state),
                ),
            ),
            dataloader_val=None,
            upload_reproducible_setup=False,
        ),
        flags={"allow_objects": True},
    )


action_humanaction_joint_nano = _make_experiment("action_humanaction_joint_nano", initial_state=None)
action_humanaction_joint_a0_nano = _make_experiment("action_humanaction_joint_a0_nano", initial_state="predict")

for _item in (action_humanaction_joint_nano, action_humanaction_joint_a0_nano):
    _cfg = _item["model"]["config"]
    # 73-frame windows (+ the shorter 1+4N snapped Mecka windows the released run compiled exactly).
    _cfg["tokenizer"]["encode_exact_durations"] = [17, 61, 73]
    # Released run: 74k-token packs per rank at 480p.
    _cfg["max_num_tokens_after_packing"] = 74000
    # Vision flow-matching loss x10 next to action_loss_weight=10 (both heads at comparable gradient magnitude).
    _cfg["rectified_flow_training_config"]["loss_scale"] = 10.0
# Image2Action: upweight the a0 row (1 of 73 rows, the hardest target) by 10 in the action loss.
action_humanaction_joint_a0_nano["model"]["config"]["rectified_flow_training_config"][
    "action_initial_state_loss_weight"
] = 10.0

for _item in (action_humanaction_joint_nano, action_humanaction_joint_a0_nano):
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
