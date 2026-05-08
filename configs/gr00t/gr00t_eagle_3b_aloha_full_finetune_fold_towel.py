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

model = dict(
    type='LlavaVLA',
    pretrained_name_or_path=  # noqa: E251
    '/limx/tos/users/liyinhao/projects/GR00T-N1.5-3B',
    vlm_backbone=dict(
        type='EagleBackbone',
        vlm_path=  # noqa: E251
        'fluxvla/models/third_party_models/eagle2_hg_model'),
    vla_head=dict(
        type='FlowMatchingHead',
        state_dim=14,
        hidden_size=1024,
        input_embedding_dim=1536,
        num_layers=1,
        num_heads=4,
        num_inference_timesteps=4,
        num_steps=32,
        traj_length=10,
        action_dim=14),
    freeze_vlm_backbone=False,
    name_mapping={
        'vlm_backbone.vlm': 'backbone.eagle_model',
        'vla_head': 'action_head'
    },
    freeze_projector=False)

inference_model = dict(
    type='LlavaVLA',
    pretrained_name_or_path=  # noqa: E251
    '/limx/tos/users/liyinhao/projects/GR00T-N1.5-3B',
    vlm_backbone=dict(
        type='EagleBackbone',
        dtype=None,
        vlm_path=  # noqa: E251
        'fluxvla/models/third_party_models/eagle2_hg_model'),
    vla_head=dict(
        type='FlowMatchingHead',
        state_dim=14,
        hidden_size=1024,
        input_embedding_dim=1536,
        num_layers=1,
        num_heads=4,
        num_steps=32,
        num_inference_timesteps=4,
        traj_length=10,
        action_dim=14),
    freeze_vlm_backbone=False,
    name_mapping={
        'vlm_backbone.vlm': 'backbone.eagle_model',
        'vla_head': 'action_head'
    },
    freeze_projector=False)

train_dataloader = dict(
    per_device_batch_size=8,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedRepeatingDataset',
        name_mappings={'observation.state': ['proprio', 'action']},
        statistic_keys=[
            'observation.state', 'observation.eepose', 'timestamp'
        ],
        datasets=[
            dict(
                type='ParquetDataset',
                data_root_path=[  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251204_20251204_01_4090_e2e/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251205_20251205_01_4090_e2e/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251205_20251205_01_4090_test_e2e/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251206_20251206_01_4090_e2e/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251208_20251208_01_4090_e2e/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251208_20251208_01_4090_test_e2e/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251209_20251209_01_4090_e2e/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251209_20251209_01_4090_test_e2e/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251210_20251210_01_4090_e2e/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251210_20251210_01_4090_test_e2e/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251210_20251210_01_4090_stage0_e2e/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251210_20251210_01_4090_test_stage0_e2e/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251211_20251211_01_4090_e2e/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251211_20251211_01_4090_e2e_dagger/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251211_20251211_01_4090_e2e_dagger_test/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251211_20251211_01_4090_e2e_test/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251212_20251212_01_4090_e2e/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251212_20251212_01_4090_test_e2e/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251213_20251213_01_4090_e2e/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251215_20251215_01_4090_e2e/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251216_20251216_01_4090_e2e/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251216_20251216_01_4090_test_e2e/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251217_20251217_01_4090_e2e/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251217_20251217_01_4090_test_e2e/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251218_20251218_01_4090_e2e/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251223_20251223_01_4090_e2e/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251224_20251224_01_4090_e2e_04/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251224_20251224_01_4090_e2e_05/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251229_20251229_01_4090_e2e/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251229_20251229_01_4090_e2e_dagger_01/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251229_20251229_01_4090_e2e_dagger_02/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251229_20251229_01_4090_e2e_dagger_03/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251229_20251229_01_4090_e2e_dagger_04/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251230_20251230_01_4090_e2e_dagger01/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251230_20251230_01_4090_e2e_dagger02/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251230_20251230_01_4090_e2e_dagger03/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251230_20251230_01_4090_e2e_dagger04/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251230_20251230_01_4090_e2e_dagger05/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251231_20251231_01_4090_e2e_dagger01/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251231_20251231_01_4090_e2e_dagger02/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251231_20251231_01_4090_e2e_dagger03/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251229_20251229_01_4090_e2e_dagger_01/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251229_20251229_01_4090_e2e_dagger_02/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251231_20251231_01_4090_e2e_dagger01/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251231_20251231_01_4090_e2e_dagger02/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251231_20251231_01_4090_e2e_dagger03/',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20260104_20260104_01_4090_e2e_01/',  # noqa: E501
                ],
                transforms=[
                    dict(
                        type='ProcessParquetInputs',
                        embodiment_id=30,
                        parquet_keys=[
                            'observation.state', 'observation.eepose',
                            'timestamp', 'actions', 'info', 'stats',
                            'action_masks'
                        ],
                        video_keys=[
                            'observation.images.cam_high',
                            'observation.images.cam_left_wrist',
                            'observation.images.cam_right_wrist'
                        ],
                        name_mappings={'observation.state': ['states']}),
                    dict(type='ParquetPrompter'),
                    dict(
                        type='ProcessPromptsWithImage',
                        max_len=900,
                        num_images=3,
                        tokenizer=dict(
                            type='PretrainedTokenizer',
                            model_path=  # noqa: E251
                            '/limx/tos/users/liyinhao/projects/eagle2_hg_model',  # noqa: E501
                            # special_tokens={'pad_token': '<PAD>'}
                        )),
                    dict(type='ResizeImages', height=224, width=224),
                    dict(
                        type='AugImage',
                        brightness_range=(0.8, 1.20),
                        prob=0.3),
                    dict(
                        type='NormalizeImages',
                        means=[[123.515625, 116.04492188, 103.59375],
                               [123.515625, 116.04492188, 103.59375],
                               [123.515625, 116.04492188, 103.59375]],
                        stds=[[58.27148438, 57.02636719, 57.27539062],
                              [58.27148438, 57.02636719, 57.27539062],
                              [58.27148438, 57.02636719, 57.27539062]],
                    ),
                    dict(
                        type='NormalizeStatesAndActions',
                        action_dim=14,
                        state_key='proprio',
                        action_key='action',
                        norm_type='mean_std')
                ],
                action_window_size=32)
        ]))

runner = dict(
    type='FSDPTrainRunner',
    max_epochs=6,
    learning_rate=2e-5,
    weight_decay=0.0,
    max_grad_norm=1.0,
    sampler=None,
    tokenizer=dict(
        type='PretrainedTokenizer',
        model_path=  # noqa: E251
        '/limx/tos/users/liyinhao/projects/eagle2_hg_model',  # noqa: E501
        # special_tokens={'pad_token': '<PAD>'}
    ),
    collator=dict(
        type='DictCollator',
        keys=[
            'states', 'observation.eepose', 'timestamp', 'images', 'img_masks',
            'lang_tokens', 'lang_masks', 'actions', 'action_masks',
            'embodiment_ids'
        ],
        meta_keys=['task_description', 'prompt', 'info', 'stats']),
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        grad_accumulation_steps=1,
        window_size=1),
    lr_scheduler_type='constant',
    warmup_ratio=0.0,
    enable_gradient_checkpointing=False,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    change_key_name=False)

inference = dict(
    type='AlohaInferenceRunner',
    seed=7,
    async_execution=False,
    postprocess_config=dict(
        enabled=True,
        method='joint_mpc',
        mode='settle',
        max_velocity=5.0,
        max_acceleration=40.0,
        max_jerk=80.0,
    ),
    task_descriptions={
        '1':
        'Fold the white towel in half, then fold it again, and make final adjustments to ensure the edges are neatly aligned.'  # noqa: E501
    },
    mixed_precision_dtype='bf16',
    dataset=dict(
        type='PrivateInferenceDataset',
        embodiment_id=30,
        img_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        transforms=[
            dict(
                type='ProcessPromptsWithImage',
                max_len=900,
                num_images=3,
                tokenizer=dict(type='PretrainedTokenizer'
                               # special_tokens={'pad_token': '<PAD>'}
                               )),
            dict(type='ResizeImages', height=224, width=224),
            dict(
                type='NormalizeImages',
                means=[[123.515625, 116.04492188, 103.59375],
                       [123.515625, 116.04492188, 103.59375],
                       [123.515625, 116.04492188, 103.59375]],
                stds=[[58.27148438, 57.02636719, 57.27539062],
                      [58.27148438, 57.02636719, 57.27539062],
                      [58.27148438, 57.02636719, 57.27539062]],
            ),
            dict(
                type='NormalizeStatesAndActions',
                state_key='proprio',
                action_key='action',
                norm_type='mean_std')
        ]),
    denormalize_action=dict(
        type='DenormalizePrivateAction', norm_type='mean_std', action_dim=14),
    prepare_pose=[[
        -0.19779752, 1.07020684, -0.61802348, -1.30887565, 1.1520192,
        2.10289164, 0.092
    ],
                  [
                      0.34008822, 0.95214585, -0.56617991, 1.13862221,
                      0.82892144, -1.80234897, 0.06909
                  ]],
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
