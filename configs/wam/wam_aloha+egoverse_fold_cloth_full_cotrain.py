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
# WAM co-training on real ALOHA fold-cloth data and Egoverse fold-cloth data.

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
        './work_dirs/wam_text_embeds_cache',
    ))

_egoverse_data_roots = [
    '/mnt/data/cpfs/users/mayer/egoverse_lerobot/fold_cloth/fold_cloth_rel/fold_cloth_relative',  # noqa: E501
]
_aloha_data_roots = [
    '/mnt/data/cpfs/users/mayer/RealRobot_AgileX_aloha_lerobot_v2/20260613_20260613_01_4090_e2e_02',  # noqa: E501
    '/mnt/data/cpfs/users/mayer/RealRobot_AgileX_aloha_lerobot_v2/20260615_20260615_01_4090_e2e_02',  # noqa: E501
]
_action_dim = 14
_proprio_dim = 14
_action_horizon = 32
_frame_window_size = 9
_frame_sample_stride = 4
_statistic_name = 'private'
_mode_probs = dict(forward=1.0, idm=1.0, policy=1.0, joint=0.0)
seed = 42
_prompt_template = (
    "A video recorded from a robot's point of view executing the following "
    'instruction: folding clothes')


def _common_transforms(
    video_keys,
    embodiment_id: int,
    tile_direction: str = 'robotwin',
    pad_missing_views: bool = False,
):
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
            video_keys=video_keys,
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
            num_views=3,
            frame_window_size=_frame_window_size,
            tile_direction=tile_direction,
            pad_missing_views=pad_missing_views,
        ),
        dict(
            type='LoadCachedTextEmbedding',
            cache_dir=_text_cache_dir,
            context_len=128,
            enc_id='wan22ti2v5b',
            prompt_template=_prompt_template,
        ),
    ]


def _wam_dataset(
    data_roots,
    video_keys,
    embodiment_id: int,
    pad_missing_views: bool = False,
):
    return dict(
        type='ParquetDataset',
        data_root_path=data_roots,
        transforms=_common_transforms(
            video_keys=video_keys,
            embodiment_id=embodiment_id,
            pad_missing_views=pad_missing_views,
        ),
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
    num_views=3,
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
            lambda_idm_action=1.0,
            lambda_policy_action=1.0,
            lambda_joint_video=0.0,
            lambda_joint_action=0.0,
        ),
        video_cond_noise_prob=0.5,
    ),
)

inference_model = dict(model, vlm_backbone=None)

train_dataloader = dict(
    per_device_batch_size=8,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedRepeatingDataset',
        seed=7,
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name=_statistic_name,
        datasets=dict(
            egoverse=[
                _wam_dataset(
                    _egoverse_data_roots,
                    video_keys=['observation.images.image'],
                    embodiment_id=5,
                    pad_missing_views=True,
                ),
            ],
            aloha=[
                _wam_dataset(
                    _aloha_data_roots,
                    video_keys=[
                        'observation.images.cam_high',
                        'observation.images.cam_left_wrist',
                        'observation.images.cam_right_wrist',
                    ],
                    embodiment_id=0,
                ),
            ],
        ),
    ),
)

runner = dict(
    type='DDPTrainRunner',
    max_epochs=5,
    optimizer=dict(
        lr=1e-4,
        type='AdamW',
        weight_decay=1e-2,
        betas=(0.9, 0.95),
    ),
    max_grad_norm=1.0,
    save_iter_interval=1000,
    max_keep_ckpts=10,
    collator=dict(
        type='WAMModeCollator',
        mode='batch',
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
        meta_keys=['task_description', 'info', 'stats', 'timestamp'],
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
    type='AlohaInferenceRunner',
    task_suite_name=_statistic_name,
    task_descriptions={
        '': 'folding clothes',
        '1': 'folding clothes',
    },
    seed=7,
    state_dim=_proprio_dim,
    action_chunk=_action_horizon,
    mixed_precision_dtype='bf16',
    dataset=dict(
        type='PrivateInferenceDataset',
        embodiment_id=0,
        img_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
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
                num_views=3,
                frame_window_size=1,
                tile_direction='robotwin',
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
        norm_type='mean_std',
        action_dim=_action_dim,
    ),
    prepare_pose=[
        [
            -0.19779752, 1.07020684, -0.61802348, -1.30887565, 1.1520192,
            2.10289164, 0.092
        ],
        [
            0.34008822, 0.95214585, -0.56617991, 1.13862221, 0.82892144,
            -1.80234897, 0.06909
        ],
    ],
    operator=dict(
        type='AlohaOperator',
        img_front_topic='/camera_h/color/image_raw',
        img_left_topic='/camera_l/color/image_raw',
        img_right_topic='/camera_r/color/image_raw',
        img_front_depth_topic='/camera_h/depth/image_raw',
        img_left_depth_topic='/camera_l/depth/image_raw',
        img_right_depth_topic='/camera_r/depth/image_raw',
        puppet_arm_left_cmd_topic='/master/joint_left',
        puppet_arm_right_cmd_topic='/master/joint_right',
        puppet_arm_left_topic='/puppet/joint_left',
        puppet_arm_right_topic='/puppet/joint_right',
        robot_base_topic='/odom_raw',
        robot_base_cmd_topic='/cmd_vel',
    ),
)
