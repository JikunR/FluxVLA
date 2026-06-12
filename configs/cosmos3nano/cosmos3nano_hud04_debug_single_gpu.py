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
# Cosmos3-Nano DEBUG config (single GPU) for rapid iteration on HUD04 data.
#
# Differences from the full finetune config:
# * Smaller image resolution (64×64) for faster VAE encoding.
# * Only 1 frame per window (no world model) → minimal memory.
# * Reduced action horizon (8 steps).
# * DDPTrainRunner instead of FSDPTrainRunner.
# * Only 2 steps max for smoke-test.
# * Only the basket dataset (single data path).
#
# Usage (1 GPU):
#   torchrun --nproc-per-node=1 scripts/train.py \
#     --config configs/cosmos3nano/cosmos3nano_hud04_debug_single_gpu.py

_ckpt_root = './checkpoints'
_cosmos3_nano_ckpt = _ckpt_root + '/Cosmos3-Nano'
_vae_path = _cosmos3_nano_ckpt + '/tokenizer'

_action_dim = 52
_max_action_dim = 64
_state_dim = 64
_action_horizon = 8
_frame_window_size = 2  # 1 conditioning + 1 future (minimal world model)
_image_height = 64
_image_width = 64

model = dict(
    type='Cosmos3VLA',
    pretrained_name_or_path=_cosmos3_nano_ckpt,
    vae_path=_vae_path,
    max_action_dim=_max_action_dim,
    action_loss_weight=10.0,
    vision_loss_weight=1.0,
    resolution='256',  # keep shift=3, resolution label not used in model
    shift=3,
    num_inference_steps=4,  # fewer steps for debug
    train_time_action_distribution='logitnormal',
    train_time_vision_distribution='waver',
    independent_action_schedule=True,
    freeze_vlm_layers=False,
    num_train_timesteps=1000,
)

train_dataloader = dict(
    per_device_batch_size=1,
    per_device_num_workers=0,  # no subprocess workers for easy debugging
    dataset=dict(
        type='DistributedRepeatingDataset',
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        datasets=dict(
            basket=[
                dict(
                    type='ParquetDataset',
                    data_root_path=[
                        '/mnt/data/cpfs/users/jikun/vcube_data/0518_basket_full_task_prompt_v4_V2.1',  # noqa: E501
                    ],
                    transforms=[
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
                                'observation.images.head',
                                'observation.images.left_wrist',
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
                            max_len=256,  # shorter for debug
                        ),
                        dict(
                            type='ResizeImages',
                            height=_image_height,
                            width=_image_width,
                        ),
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
                            embodiment_to_domain_id={
                                0: 8,
                                1: 8
                            },
                            mode='policy',
                            frame_window_size=_frame_window_size,
                            num_conditioning_vision_frames=1,
                            conditioning_fps=15.0,
                        ),
                        dict(
                            type='PrepareVideo',
                            num_views=2,
                            frame_window_size=_frame_window_size,
                        ),
                    ],
                    action_window_size=_action_horizon,
                    action_key='action',
                    use_delta=False,
                    window_start_idx=0,
                    frame_window_size=_frame_window_size,
                )
            ], ),
    ),
)

runner = dict(
    type='FSDPTrainRunner',
    max_epochs=1,
    max_steps=2,  # smoke-test: 2 gradient steps then exit
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
        active_trackers=('jsonl', ),  # no wandb for debug
        run_dir='work_dirs',
        grad_accumulation_steps=1,
        window_size=1,
    ),
    lr_scheduler_type='constant',
    warmup_ratio=0.0,
    enable_gradient_checkpointing=False,  # disable for easier debugging
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    sharding_strategy='no-shard',  # easier to debug than full-shard
    change_key_name=False,
)
