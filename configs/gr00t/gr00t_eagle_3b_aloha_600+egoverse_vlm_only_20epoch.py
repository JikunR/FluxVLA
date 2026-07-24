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
# GR00T Eagle co-training on real ALOHA fold-cloth data and Egoverse
# fold-cloth data. This config loads the unfine-tuned Eagle VLM checkpoint
# only; the action head is randomly initialized and trained from scratch.

_eagle_base_model_id = ('nvidia/Eagle2-2B')
_eagle_vlm_ckpt = './checkpoints/Eagle2-2B'
_egoverse_data_roots = [
    '/mnt/data/cpfs/users/mayer/egoverse_lerobot/fold_cloth/fold_cloth_rel/fold_cloth_relative',  # noqa: E501
]
_aloha_data_roots = [
    '/mnt/data/cpfs/users/mayer/RealRobot_AgileX_aloha_lerobot_v2/20260613_20260613_01_4090_e2e_02',  # noqa: E501
    '/mnt/data/cpfs/users/mayer/RealRobot_AgileX_aloha_lerobot_v2/20260615_20260615_01_4090_e2e_02',  # noqa: E501
]
_action_dim = 14
_proprio_dim = 14
_state_dim = 64
_head_action_dim = 32
_vlm_hidden_dim = 1536
_action_horizon = 32
_statistic_name = 'private'
_vlm_only_name_mapping = {'vlm_backbone.vlm.': ''}


def _common_transforms(video_keys, embodiment_id: int):
    return [
        dict(
            type='ProcessParquetInputs',
            embodiment_id=embodiment_id,
            parquet_keys=[
                'observation.state',
                'timestamp',
                'actions',
                'info',
                'stats',
                'action_masks',
            ],
            video_keys=video_keys,
            name_mappings={'observation.state': ['states']}),
        dict(type='ParquetPrompter'),
        dict(
            type='ProcessPromptsWithImage',
            max_len=900,
            num_images=3,
            tokenizer=dict(
                type='PretrainedTokenizer',
                model_path=_eagle_vlm_ckpt,
            )),
        dict(type='ResizeImages', height=448, width=448),
        dict(
            type='NormalizeImages',
            means=[
                [123.515625, 116.04492188, 103.59375],
                [123.515625, 116.04492188, 103.59375],
                [123.515625, 116.04492188, 103.59375],
            ],
            stds=[
                [58.27148438, 57.02636719, 57.27539062],
                [58.27148438, 57.02636719, 57.27539062],
                [58.27148438, 57.02636719, 57.27539062],
            ],
        ),
        dict(
            type='NormalizeStatesAndActions',
            state_dim=_state_dim,
            action_dim=_head_action_dim,
            state_key='proprio',
            action_key='action',
            norm_type='mean_std')
    ]


def _gr00t_dataset(data_roots, video_keys, embodiment_id: int):
    return dict(
        type='ParquetDataset',
        data_root_path=data_roots,
        transforms=_common_transforms(
            video_keys=video_keys,
            embodiment_id=embodiment_id,
        ),
        action_window_size=_action_horizon,
        action_key='action',
        window_start_idx=0,
        statistic_name=_statistic_name)


model = dict(
    type='LlavaVLA',
    pretrained_name_or_path=_eagle_vlm_ckpt,
    vlm_backbone=dict(
        type='EagleBackbone',
        vlm_path=_eagle_vlm_ckpt,
        vlm_config=dict(max_input_seq_len=900)),
    vla_head=dict(
        type='FlowMatchingHead',
        state_dim=_state_dim,
        hidden_size=1024,
        input_embedding_dim=_vlm_hidden_dim,
        backbone_embedding_dim=_vlm_hidden_dim,
        vl_self_attention_cfg=dict(
            attention_head_dim=64,
            dropout=0.2,
            final_dropout=True,
            num_attention_heads=24,
            num_layers=4,
            positional_embeddings=None),
        diffusion_model_cfg=dict(
            attention_head_dim=48,
            cross_attention_dim=_vlm_hidden_dim,
            dropout=0.2,
            final_dropout=True,
            interleave_self_attention=True,
            norm_type='ada_norm',
            num_attention_heads=32,
            num_layers=16,
            output_dim=1024,
            positional_embeddings=None),
        num_inference_timesteps=4,
        num_steps=32,
        action_dim=_head_action_dim,
        ori_action_dim=_action_dim),
    freeze_vlm_backbone=False,
    name_mapping=_vlm_only_name_mapping,
    freeze_projector=False)

inference_model = dict(
    type='LlavaVLA',
    pretrained_name_or_path=_eagle_vlm_ckpt,
    vlm_backbone=dict(
        type='EagleBackbone',
        vlm_path=_eagle_vlm_ckpt,
        vlm_config=dict(max_input_seq_len=900)),
    vla_head=dict(
        type='FlowMatchingInferenceHead',
        state_dim=_state_dim,
        hidden_size=1024,
        input_embedding_dim=_vlm_hidden_dim,
        backbone_embedding_dim=_vlm_hidden_dim,
        vl_self_attention_cfg=dict(
            attention_head_dim=64,
            dropout=0.2,
            final_dropout=True,
            num_attention_heads=24,
            num_layers=4,
            positional_embeddings=None),
        num_steps=32,
        num_inference_timesteps=4,
        ori_action_dim=_action_dim,
        action_dim=_head_action_dim,
        max_input_seq_len=900,
        diffusion_model_cfg=dict(
            attention_head_dim=48,
            cross_attention_dim=_vlm_hidden_dim,
            dropout=0.2,
            final_dropout=True,
            interleave_self_attention=True,
            norm_type='ada_norm',
            num_attention_heads=32,
            num_layers=16,
            output_dim=1024,
            positional_embeddings=None)),
    freeze_vlm_backbone=False,
    name_mapping=_vlm_only_name_mapping,
    freeze_projector=False)

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
        statistic_keys=[
            'observation.state',
            'timestamp',
            'action',
        ],
        statistic_name=_statistic_name,
        datasets=dict(
            egoverse=[
                _gr00t_dataset(
                    _egoverse_data_roots,
                    video_keys=[
                        'observation.images.image',
                        'observation.images.image',
                        'observation.images.image',
                    ],
                    embodiment_id=5,
                ),
            ],
            aloha=[
                _gr00t_dataset(
                    _aloha_data_roots,
                    video_keys=[
                        'observation.images.cam_high',
                        'observation.images.cam_left_wrist',
                        'observation.images.cam_right_wrist',
                    ],
                    embodiment_id=0,
                ),
            ],
        )))

runner = dict(
    type='FSDPTrainRunner',
    max_epochs=20,
    optimizer=dict(lr=2e-5, type='AdamW', weight_decay=0.0),
    max_grad_norm=1.0,
    sampler=None,
    tokenizer=dict(
        type='PretrainedTokenizer',
        model_path=_eagle_vlm_ckpt,
    ),
    collator=dict(
        type='DictCollator',
        keys=[
            'states',
            'timestamp',
            'images',
            'img_masks',
            'lang_tokens',
            'lang_masks',
            'actions',
            'action_masks',
            'embodiment_ids',
        ],
        meta_keys=['task_description', 'prompt', 'info', 'stats']),
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        grad_accumulation_steps=1,
        window_size=1),
    lr_scheduler=dict(type='constant'),
    enable_gradient_checkpointing=False,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    change_key_name=False)

inference = dict(
    type='AlohaInferenceRunner',
    task_suite_name=_statistic_name,
    seed=7,
    task_descriptions={
        '': 'folding clothes',
        '1': 'folding clothes',
    },
    state_dim=_proprio_dim,
    action_chunk=_action_horizon,
    mixed_precision_dtype='bf16',
    dataset=dict(
        type='PrivateInferenceDataset',
        embodiment_id=0,
        img_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        transforms=[
            dict(
                type='ProcessPromptsWithImage',
                max_len=900,
                num_images=3,
                tokenizer=dict(type='PretrainedTokenizer')),
            dict(type='ResizeImages', height=448, width=448),
            dict(
                type='NormalizeImages',
                means=[
                    [123.515625, 116.04492188, 103.59375],
                    [123.515625, 116.04492188, 103.59375],
                    [123.515625, 116.04492188, 103.59375],
                ],
                stds=[
                    [58.27148438, 57.02636719, 57.27539062],
                    [58.27148438, 57.02636719, 57.27539062],
                    [58.27148438, 57.02636719, 57.27539062],
                ],
            ),
            dict(
                type='NormalizeStatesAndActions',
                state_dim=_state_dim,
                action_dim=_head_action_dim,
                state_key='proprio',
                action_key='action',
                norm_type='mean_std')
        ]),
    denormalize_action=dict(
        type='DenormalizePrivateAction',
        norm_type='mean_std',
        action_dim=_action_dim),
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
    ))
