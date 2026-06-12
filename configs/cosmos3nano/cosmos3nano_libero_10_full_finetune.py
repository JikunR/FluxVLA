# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Cosmos3-Nano full finetune on LIBERO-10 benchmark.
#
# Usage (8 GPU):
#   torchrun --nproc-per-node=8 scripts/train.py \
#     --config configs/cosmos3nano/cosmos3nano_libero_10_full_finetune.py
#
# Key differences from HUD04 config:
# * LIBERO: action_dim=7 (eef pos/rot + gripper), state_dim=32, 2 views.
# * Video keys: observation.images.image + observation.images.wrist_image.
# * Single dataset group (no multi-embodiment split).
# * domain_id=0 (no_action default; fine-tuning will adapt the domain head).
# * 128×128 image resolution (same as other VLAs on LIBERO for fair
#   comparison).

_ckpt_root = './checkpoints'
_cosmos3_nano_ckpt = _ckpt_root + '/Cosmos3-Nano'
_vae_path = _cosmos3_nano_ckpt + '/tokenizer'

# LIBERO robot spec
_action_dim = 7  # eef_pos(3) + eef_ori(3) + gripper(1)
_max_action_dim = 32  # padded to match libero convention (same as dreamzero)
_state_dim = 32  # eef_pos(3) + eef_quat(4) + gripper(2) + ... padded to 32
_action_horizon = 10  # matches dreamzero libero setting
_frame_window_size = 5  # 1 conditioning + 4 future frames
_image_height = 128
_image_width = 128  # each view; two views stacked → 128×256

model = dict(
    type='Cosmos3VLA',
    pretrained_name_or_path=_cosmos3_nano_ckpt,
    vae_path=_vae_path,
    max_action_dim=_max_action_dim,
    action_loss_weight=10.0,
    vision_loss_weight=1.0,
    resolution='256',
    shift=3,
    num_inference_steps=20,
    train_time_action_distribution='logitnormal',
    train_time_vision_distribution='waver',
    independent_action_schedule=True,
    freeze_vlm_layers=False,  # full finetune
    num_train_timesteps=1000,
)

_transforms = [
    dict(
        type='ProcessParquetInputs',
        parquet_keys=[
            'observation.state',
            'timestamp',
            'actions',
            'info',
            'stats',
            'action_masks',
        ],
        video_keys=[
            'observation.images.image',
            'observation.images.wrist_image',
        ],
        name_mappings={
            'observation.state': ['states'],
            'actions': ['actions'],
        },
        embodiment_id=0,
    ),
    dict(type='ParquetPrompter', use_conversation=False),
    dict(
        type='ProcessCosmos3Prompt',
        qwen3_vl_model_path=_cosmos3_nano_ckpt,
        max_len=512,
    ),
    dict(type='ResizeImages', height=_image_height, width=_image_width),
    # SimpleNormalizeImages: scales [0,255] uint8 → [-1,1] float32
    # (required by Wan2.2 VAE)
    dict(type='SimpleNormalizeImages'),
    dict(
        type='NormalizeStatesAndActions',
        action_dim=_max_action_dim,
        state_dim=_state_dim,
        state_key='proprio',
        action_key='action',
        norm_type='mean_std',
    ),
    dict(
        type='BuildCosmos3Sequence',
        max_action_dim=_max_action_dim,
        # LIBERO has a single embodiment; use domain_id=0 (no_action default).
        # The DomainAwareLinear head will be fine-tuned to this domain.
        embodiment_to_domain_id={0: 0},
        mode='policy',
        frame_window_size=_frame_window_size,
        num_conditioning_vision_frames=1,
        conditioning_fps=20.0,  # LIBERO is recorded at ~20 fps
    ),
    dict(
        type='PrepareVideo',
        num_views=2,
        frame_window_size=_frame_window_size,
    ),
]

train_dataloader = dict(
    per_device_batch_size=2,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedRepeatingDataset',
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name='libero_10_no_noops',
        datasets=dict(
            type='ParquetDataset',
            data_root_path='./datasets/libero_10_no_noops_lerobotv2.1',
            transforms=_transforms,
            action_window_size=_action_horizon,
            action_key='action',
            use_delta=False,
            statistic_name='libero_10_no_noops',
            window_start_idx=0,
            frame_window_size=_frame_window_size,
        ),
    ),
)

runner = dict(
    type='FSDPTrainRunner',
    max_epochs=12,
    learning_rate=2e-5,
    weight_decay=0.0,
    max_grad_norm=0.1,
    collator=dict(
        type='Cosmos3Collator',
        tensor_keys=[
            'images',
            'actions',
            'domain_id',
            'raw_action_dim',
            'conditioning_fps',
        ],
        sequence_keys=['text_token_ids'],
        list_keys=['sequence_plan'],
        meta_keys=['task_description', 'stats', 'info', 'timestamp'],
        pad_id=0,
    ),
    sampler=None,
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        grad_accumulation_steps=1,
        window_size=1,
    ),
    lr_scheduler_type='linear-warmup+cosine-decay',
    warmup_ratio=0.05,
    enable_gradient_checkpointing=True,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    sharding_strategy='full-shard',
    change_key_name=False,
)

eval = dict(
    type='LiberoEvalRunner',
    task_suite_name='libero_10',
    model_family='cosmos3',
    eval_chunk_size=10,
    resize_size=128,
    num_trials_per_task=50,
    num_steps_wait=10,
    seed=7,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    dataset=dict(
        type='LiberoParquetEvalDataset',
        img_buffer_len=1,
        transforms=[
            dict(
                type='ProcessLiberoEvalInputs',
                img_keys=['agentview_image', 'robot0_eye_in_hand_image'],
            ),
            dict(
                type='TransformImage',
                image_resize_strategy='resize-naive',
                input_sizes=[[3, _image_height, _image_width],
                             [3, _image_height, _image_width]],
                means=[[127.5, 127.5, 127.5], [127.5, 127.5, 127.5]],
                stds=[[127.5, 127.5, 127.5], [127.5, 127.5, 127.5]],
            ),
            dict(
                type='LiberoProprioFromInputs',
                norm_type='mean_std',
                pos_key='robot0_eef_pos',
                quat_key='robot0_eef_quat',
                gripper_key='robot0_gripper_qpos',
                state_dim=_state_dim,
                out_key='states',
            ),
            dict(
                type='LiberoPromptFromInputs',
                tokenizer=dict(
                    type='PretrainedTokenizer',
                    model_path=_cosmos3_nano_ckpt,
                ),
                max_len=512,
                use_conversation=False,
            ),
            dict(
                type='PrepareVideo',
                num_views=2,
                frame_window_size=1,
            ),
        ],
    ),
    denormalize_action=dict(
        type='DenormalizeLiberoAction',
        norm_type='mean_std',
        action_dim=_action_dim,
    ),
)
