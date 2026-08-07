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
# WAM on the HUD04 / VCube basket data with the Wan2.2 UMT5 text encoder.
# One MoT forward jointly trains video generation (VGM/forward_video) and
# policy action prediction via the FastWAM-style ``vgm_policy`` mode.

import os

_repo_root = os.path.abspath(os.environ.get('FLUXVLA_ROOT', '.'))
_ckpt_root = os.path.join(_repo_root, 'checkpoints')
_wan_checkpoint_root = os.path.abspath(
    os.environ.get('WAN22_CHECKPOINT_ROOT',
                   os.path.join(_ckpt_root, 'Wan2.2-TI2V-5B')))
_action_dit = os.path.abspath(
    os.environ.get(
        'ACTION_DIT_PATH',
        os.path.join(_ckpt_root,
                     'ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt')))
_text_cache_dir = os.path.abspath(
    os.environ.get(
        'WAM_TEXT_CACHE_DIR',
        os.path.join(_ckpt_root, 'hud04', 'text_embeds_cache'),
    ))

_data_root = '/mnt/data/cpfs/users/jikun/vcube_data'
_basket_data_roots = [
    os.path.join(
        _data_root,
        '0518_21_22_25_26_27_0601_02_03_04_05_09_10_11_12_15_16_17_18_22_23_basket_full_task_prompt_delta_base_filtered_V2.1',  # noqa: E501
    )
]
_action_dim = 64
_proprio_dim = 64
_action_horizon = 32
_frame_window_size = 9
_frame_sample_stride = 4
_statistic_name = 'hud04_vcube'
_mode_probs = dict(
    forward=0.0,
    idm=0.0,
    policy=0.0,
    joint=0.0,
    vgm_policy=1.0,
)
seed = 42
_prompt_template = (
    "A video recorded from a robot's point of view executing the following "
    'instruction: {task}')
_task_prompt = (
    'Lift up the red basket with right arm, put all the objects on the white '
    'table into the red basket with left arm, place the red basket on the '
    'table.')


def _vcube_pipeline(embodiment_id: int):
    return [
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
            embodiment_id=embodiment_id,
        ),
        dict(
            type='ResizeImages',
            height=240,
            width=320,
        ),
        dict(
            type='NormalizeImages',
            means=[0.5, 0.5, 0.5],
            stds=[0.5, 0.5, 0.5],
            scale_to_unit_interval=True,
        ),
        dict(
            type='NormalizeStatesAndActions',
            action_dim=_action_dim,
            state_dim=_proprio_dim,
            state_key='proprio',
            action_key='action',
            norm_type='mean_std',
        ),
        dict(
            type='PrepareVideo',
            num_views=2,
            frame_window_size=_frame_window_size,
            tile_direction='vertical',
        ),
        dict(
            type='LoadCachedTextEmbedding',
            cache_dir=_text_cache_dir,
            context_len=128,
            enc_id='wan22ti2v5b',
            prompt_template=_prompt_template,
        ),
    ]


def _vcube_dataset(data_roots, embodiment_id: int):
    return dict(
        type='ParquetDataset',
        data_root_path=data_roots,
        transforms=_vcube_pipeline(embodiment_id),
        action_window_size=_action_horizon,
        action_key='action',
        use_delta=False,
        statistic_name=_statistic_name,
        window_start_idx=0,
        frame_window_size=_frame_window_size,
        frame_sample_stride=_frame_sample_stride,
    )


model = dict(
    type='WAMVLA',
    pretrained_name_or_path=None,
    num_views=2,
    frame_window_size=_frame_window_size,
    proprio_dim=_proprio_dim,
    action_horizon=_action_horizon,
    mot_checkpoint_mixed_attn=True,
    vlm_backbone=None,
    video_latent_codec=dict(
        type='Wan22VAE',
        checkpoint_root=_wan_checkpoint_root,
    ),
    vla_head=dict(
        type='WAMHead',
        video_expert=dict(
            type='WanVideoDiT',
            checkpoint_root=_wan_checkpoint_root,
            skip_load_from_pretrain=False,
            config=dict(
                has_image_input=False,
                patch_size=[1, 2, 2],
                in_dim=48,
                hidden_dim=3072,
                ffn_dim=14336,
                freq_dim=256,
                text_dim=4096,
                out_dim=48,
                num_heads=24,
                attn_head_dim=128,
                num_layers=30,
                eps=1.0e-06,
                seperated_timestep=True,
                require_clip_embedding=False,
                require_vae_embedding=False,
                fuse_vae_embedding_in_latents=True,
                video_attention_mask_mode='first_frame_causal',
                action_conditioned=False,
                action_dim=_action_dim,
                action_group_causal_mask_mode='group_diagonal',
                use_gradient_checkpointing=True,
            ),
        ),
        action_expert=dict(
            type='ActionDiT',
            pretrained_path=_action_dit,
            skip_load_from_pretrain=False,
            config=dict(
                action_dim=_action_dim,
                hidden_dim=1024,
                ffn_dim=4096,
                num_heads=24,
                attn_head_dim=128,
                num_layers=30,
                text_dim=4096,
                freq_dim=256,
                eps=1.0e-06,
                use_gradient_checkpointing=True,
            ),
        ),
        video_scheduler=dict(
            train_shift=5.0, infer_shift=5.0, num_train_timesteps=1000),
        action_scheduler=dict(
            train_shift=5.0, infer_shift=5.0, num_train_timesteps=1000),
        loss=dict(
            lambda_video=0.0,
            lambda_action=0.0,
            lambda_forward_video=1.0,
            lambda_idm_action=0.0,
            lambda_policy_action=1.0,
            lambda_joint_video=0.0,
            lambda_joint_action=0.0,
        ),
        video_cond_noise_prob=0.5,
    ),
)

inference_model = dict(**model, skip_load=True)

train_dataloader = dict(
    per_device_batch_size=8,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedRepeatingDataset',
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name=_statistic_name,
        datasets=_vcube_dataset(_basket_data_roots, embodiment_id=0),
    ),
)

runner = dict(
    type='DDPTrainRunner',
    max_epochs=6,
    optimizer=dict(
        lr=1e-4,
        type='AdamW',
        weight_decay=1e-2,
        betas=(0.9, 0.95),
    ),
    max_grad_norm=1.0,
    collator=dict(
        type='WAMModeCollator',
        mode='vgm_policy',
        mode_probs=_mode_probs,
        keys=[
            'states',
            'images',
            'img_masks',
            'actions',
            'action_masks',
            'embodiment_ids',
            'frame_masks',
            'context',
            'context_mask',
            'training_mode',
        ],
        meta_keys=['task_description', 'prompt', 'info', 'stats', 'timestamp'],
    ),
    sampler=None,
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        window_size=1,
    ),
    lr_scheduler=dict(type='linear-warmup+cosine-decay', warmup_ratio=0.05),
    enable_gradient_checkpointing=False,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
)

inference = dict(
    type='OliInferenceRunner',
    task_suite_name=_statistic_name,
    task_descriptions={
        '1': _task_prompt,
    },
    seed=7,
    state_dim=_proprio_dim,
    action_chunk=_action_horizon,
    publish_rate=30,
    mixed_precision_dtype='bf16',
    low_cpu_mem_usage=True,
    camera_names=['head', 'left_wrist'],
    dataset=dict(
        type='PrivateInferenceDataset',
        statistic_name=_statistic_name,
        embodiment_id=0,
        img_keys=['head', 'left_wrist'],
        transforms=[
            dict(type='ResizeImages', height=240, width=320),
            dict(
                type='NormalizeImages',
                means=[0.5, 0.5, 0.5],
                stds=[0.5, 0.5, 0.5],
                scale_to_unit_interval=True,
            ),
            dict(
                type='NormalizeStatesAndActions',
                action_dim=_action_dim,
                state_dim=_proprio_dim,
                state_key='proprio',
                action_key='action',
                norm_type='mean_std',
            ),
            dict(
                type='PrepareVideo',
                num_views=2,
                frame_window_size=1,
                tile_direction='vertical',
            ),
            dict(
                type='LoadCachedTextEmbedding',
                cache_dir=_text_cache_dir,
                context_len=128,
                enc_id='wan22ti2v5b',
                prompt_template=_prompt_template,
            ),
        ],
    ),
    denormalize_action=dict(
        type='DenormalizePrivateAction',
        statistic_name=_statistic_name,
        norm_type='mean_std',
        action_dim=42,
    ),
    operator=dict(
        type='MrosOliOperator',
        head_rgb_topic='/head/color/image_raw/compressed',
        left_wrist_rgb_topic=(
            '/left_wrist_camera/color/image_raw/compressed'),
        joint_state_topic='/joint/state',
        finger_state_topic='/brainco1/hand/state',
        finger_cmd_topic='/brainco1/hand/cmd_vla',
        teleop_wbt_topic='/teleop_cmd_WBT',
    ),
)
