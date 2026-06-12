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
# Cosmos3-Nano full finetune on HUD04 robot data (basket + candy tasks).
#
# Usage (8 GPU):
#   torchrun --nproc-per-node=8 scripts/train.py \
#     --config configs/cosmos3nano/cosmos3nano_hud04_full_finetune.py
#
# Key design choices vs. DreamZero HUD04:
# * Uses Cosmos3-Nano (Qwen3-VL-8B MoT) backbone instead of Wan2.1-14B DiT.
# * Text prompts are tokenised by the Qwen3-VL tokenizer (plain-text,
#   no image placeholders) via ProcessCosmos3NanoPrompt.
# * Video is kept as raw RGB in [-1,1] (SimpleNormalizeImages) and encoded
#   by the Wan2.2 VAE inside the model forward pass – not in the dataset.
# * Actions and video are jointly denoised via Rectified Flow through the
#   same MoT Transformer.
# * domain_id=8 maps both embodiment groups to the droid_lerobot domain.

_ckpt_root = './checkpoints'
_cosmos3_nano_ckpt = _ckpt_root + '/Cosmos3-Nano'
# vae_path: the Wan2.2 VAE weight file inside the downloaded checkpoint.
# This must always point to the *original backbone* directory even when
# resuming, because the VAE is frozen and not saved in FluxVLA checkpoints.
_vae_path = _cosmos3_nano_ckpt + '/tokenizer'

# HUD04 robot spec
_action_dim = 52  # true (unpadded) HUD04 action dimension
_max_action_dim = 64  # padded dimension (matching cosmos3-nano config)
_state_dim = 64
_action_horizon = 24  # number of action steps predicted per inference
_frame_window_size = 5  # total video frames per training window
#                         # (1 conditioning + 4 future frames)
_image_height = 256
_image_width = 256  # each view; two views are stacked → 256×512 concat

model = dict(
    type='Cosmos3NanoVLA',
    pretrained_name_or_path=_cosmos3_nano_ckpt,
    vae_path=_vae_path,
    max_action_dim=_max_action_dim,
    action_loss_weight=10.0,
    vision_loss_weight=1.0,
    resolution='256',
    shift=3,  # RF shift for 256px resolution
    num_inference_steps=20,
    train_time_action_distribution='logitnormal',
    train_time_vision_distribution='waver',
    independent_action_schedule=True,
    freeze_vlm_layers=False,  # full finetune
    num_train_timesteps=1000,
)

_transforms_basket = [
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
        type='ProcessCosmos3NanoPrompt',
        qwen3_vl_model_path=_cosmos3_nano_ckpt,
        max_len=4096,
    ),
    dict(type='ResizeImages', height=_image_height, width=_image_width),
    # SimpleNormalizeImages: scales [0,255] uint8 → [-1,1] float32
    # (required by Wan2.2 VAE, unlike Eagle NormalizeImages)
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
        type='BuildCosmos3NanoSequence',
        max_action_dim=_max_action_dim,
        # basket=0, candy=1 → both map to droid_lerobot domain 8
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
]

_transforms_candy = [
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
        embodiment_id=1,
    ),
    dict(type='ParquetPrompter', use_conversation=False),
    dict(
        type='ProcessCosmos3NanoPrompt',
        qwen3_vl_model_path=_cosmos3_nano_ckpt,
        max_len=4096,
    ),
    dict(type='ResizeImages', height=_image_height, width=_image_width),
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
        type='BuildCosmos3NanoSequence',
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
]

train_dataloader = dict(
    per_device_batch_size=1,
    per_device_num_workers=4,
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
                        '/mnt/data/cpfs/users/jikun/vcube_data/0521_basket_full_task_prompt_v4_V2.1',  # noqa: E501
                        '/mnt/data/cpfs/users/jikun/vcube_data/0522_basket_full_task_prompt_v4_V2.1',  # noqa: E501
                        '/mnt/data/cpfs/users/jikun/vcube_data/0525_basket_full_task_prompt_v4_V2.1',  # noqa: E501
                        '/mnt/data/cpfs/users/jikun/vcube_data/0526_basket_full_task_prompt_v4_V2.1',  # noqa: E501
                        '/mnt/data/cpfs/users/jikun/vcube_data/0527_basket_full_task_prompt_v4_V2.1',  # noqa: E501
                        '/mnt/data/cpfs/users/jikun/vcube_data/0601_basket_full_task_prompt_v4_V2.1',  # noqa: E501
                        '/mnt/data/cpfs/users/jikun/vcube_data/0602_basket_full_task_prompt_v4_V2.1',  # noqa: E501
                        '/mnt/data/cpfs/users/jikun/vcube_data/0603_basket_full_task_prompt_v4_V2.1',  # noqa: E501
                        '/mnt/data/cpfs/users/jikun/vcube_data/0604_basket_full_task_prompt_v4_V2.1',  # noqa: E501
                        '/mnt/data/cpfs/users/jikun/vcube_data/0605_basket_full_task_prompt_v4_V2.1',  # noqa: E501
                    ],
                    transforms=_transforms_basket,
                    action_window_size=_action_horizon,
                    action_key='action',
                    use_delta=False,
                    window_start_idx=0,
                    frame_window_size=_frame_window_size,
                )
            ],
            candy=[
                dict(
                    type='ParquetDataset',
                    data_root_path=[
                        '/mnt/data/cpfs/users/jikun/vcube_data/0528_candy_full_task_prompt_v4_V2.1',  # noqa: E501
                        '/mnt/data/cpfs/users/jikun/vcube_data/0529_candy_full_task_prompt_v4_V2.1',  # noqa: E501
                        '/mnt/data/cpfs/users/jikun/vcube_data/0601_candy_full_task_prompt_v4_V2.1',  # noqa: E501
                        '/mnt/data/cpfs/users/jikun/vcube_data/0602_candy_full_task_prompt_v4_V2.1',  # noqa: E501
                        '/mnt/data/cpfs/users/jikun/vcube_data/0603_candy_full_task_prompt_v4_V2.1',  # noqa: E501
                        '/mnt/data/cpfs/users/jikun/vcube_data/0604_candy_full_task_prompt_v4_V2.1',  # noqa: E501
                        '/mnt/data/cpfs/users/jikun/vcube_data/0605_candy_full_task_prompt_v4_V2.1',  # noqa: E501
                    ],
                    transforms=_transforms_candy,
                    action_window_size=_action_horizon,
                    action_key='action',
                    use_delta=False,
                    window_start_idx=0,
                    frame_window_size=_frame_window_size,
                )
            ],
        ),
    ),
)

runner = dict(
    type='FSDPTrainRunner',
    max_epochs=2,
    learning_rate=2e-5,
    weight_decay=0.0,
    max_grad_norm=0.1,
    collator=dict(
        type='Cosmos3NanoCollator',
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

inference = dict(
    type='Cosmos3NanoInferenceRunner',
    seed=7,
    action_dim=_action_dim,
    action_horizon=_action_horizon,
    task_descriptions={
        '1':
        'Lift up the red basket with right arm, put all the objects on '
        'the white table into the red basket with left arm, place the '
        'red basket on the table.',
        '2':
        'Grasp the cup and pour the candies onto the table with the '
        'right arm, hang the mug on the mug rack with the right arm, '
        'and classify all candies on the table into the snack tray with '
        'the left arm.',
    },
    mixed_precision_dtype='bf16',
    dataset=dict(
        type='Cosmos3NanoInferenceDataset',
        norm_stats='',  # path to episodes_stats.jsonl or norm_stats.json
        qwen3_vl_model_path=_cosmos3_nano_ckpt,
        img_keys=['head', 'left_wrist'],
        embodiment_id=0,
        domain_id=8,
        max_action_dim=_max_action_dim,
        action_horizon=_action_horizon,
        frame_window_size=_frame_window_size,
        transforms=[
            dict(
                type='ResizeImages', height=_image_height, width=_image_width),
            dict(type='SimpleNormalizeImages'),
            dict(
                type='NormalizeStatesAndActions',
                action_dim=_max_action_dim,
                state_dim=_state_dim,
                state_key='proprio',
                action_key='action',
                norm_type='mean_std',
            ),
        ],
    ),
    denormalize_action=dict(
        type='DenormalizePrivateAction',
        norm_type='mean_std',
        action_dim=_action_dim,
    ),
    operator=dict(
        type='Teleop02WbtOperator',
        head_rgb_topic='/head/color/image_raw/compressed',
        left_wrist_rgb_topic='/left_wrist_camera/color/image_raw/compressed',
        joint_state_topic='/joint/state',
        finger_state_topic='/brainco1/hand/state',
        finger_cmd_topic='/brainco1/hand/cmd',
        teleop_wbt_topic='/teleop_cmd_WBT',
        cmd_vel_topic='/sdk_cmd_vel_vla',
    ),
)
