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

inference_model = dict(
    type='LlavaVLA',
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
                data_root_path=[
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251111_20251111_01_4090_e2e',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251112_20251112_01_4090_e2e',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251113_20251113_01_4090_e2e',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251115_20251115_01_4090_e2e',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251114_20251114_01_4090_e2e',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251117_20251117_01_4090_e2e',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/'
                    'RealRobot_AgileX_aloha_lerobot/20251118_20251118_01_4090_e2e',  # noqa: E501
                ],
                transforms=[
                    dict[str, str | int | list[str] | dict[str, list[str]]](
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
                        name_mappings={
                            'observation.state': ['states']
                        }),
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
    max_epochs=3,
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
        wandb_project='fluxvla',
        wandb_entity='limx',
        grad_accumulation_steps=1,
        window_size=1),
    lr_scheduler_type='constant',
    warmup_ratio=0.0,
    enable_gradient_checkpointing=False,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    sharding_strategy='full-shard',
    change_key_name=False)

inference = dict(
    type='AlohaInferenceRunner',
    seed=7,
    task_descriptions={
        '1':
        'Fold the white towel in half, then fold it again, and make final adjustments to ensure the edges are neatly aligned.'  # noqa: E501
    },
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
                state_dim=14,
                state_key='proprio',
                action_key='action',
                norm_type='mean_std')
        ]),
    denormalize_action=dict(
        type='DenormalizePrivateAction',
        norm_type='mean_std',
    ),
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
