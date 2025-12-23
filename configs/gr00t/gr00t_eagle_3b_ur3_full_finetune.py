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
        state_dim=7,
        hidden_size=1024,
        input_embedding_dim=1536,
        num_layers=1,
        num_heads=4,
        num_inference_timesteps=4,
        traj_length=10,
        action_dim=7),
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
        state_dim=7,
        hidden_size=1024,
        input_embedding_dim=1536,
        num_layers=1,
        num_heads=4,
        num_inference_timesteps=4,
        traj_length=10,
        action_dim=7),
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
                data_root_path=  # noqa: E251
                [
                    '/limx/tos/limx_mani_data/raw_data/RealRobot_UR3_lerobot_v2/20250501_20250531_01',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/RealRobot_UR3_lerobot_v2/20250601_20250607_01',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/RealRobot_UR3_lerobot_v2/20250608_20250611_01',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/RealRobot_UR3_lerobot_v2/20250612_20250615_01',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/RealRobot_UR3_lerobot_v2/20250616_20250815_01',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/RealRobot_UR3_lerobot_v2/20250816_20250822_01',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/RealRobot_UR3_lerobot_v2/20250823_20250831_01',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/RealRobot_UR3_lerobot_v2/20250901_20250907_01',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/RealRobot_UR3_lerobot_v2/20250908_20250915_01',  # noqa: E501
                    '/limx/tos/limx_mani_data/raw_data/RealRobot_UR3_lerobot_v2/20250916_20250930_01',  # noqa: E501
                ],
                transforms=[
                    dict(
                        type='ProcessParquetInputs',
                        embodiment_id=31,
                        parquet_keys=[
                            'observation.state', 'observation.eepose',
                            'timestamp', 'actions', 'info', 'stats',
                            'action_masks'
                        ],
                        video_keys=[
                            'observation.images.cam_high',
                            'observation.images.cam_wrist'
                        ],
                        name_mappings={'observation.state': ['states']}),
                    dict(type='ParquetPrompter'),
                    dict(
                        type='ProcessPromptsWithImage',
                        max_len=600,
                        num_images=2,
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
                               [123.515625, 116.04492188, 103.59375]],
                        stds=[[58.27148438, 57.02636719, 57.27539062],
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
    type='URInferenceRunner',
    seed=7,
    action_chunk=32,
    task_descriptions={
        '1': 'pick up the shanghai green',
        '2': 'put the shanghai green into the bamboo basket',
        '3': 'put the apple into the gray plate',
        '4': 'pick up the onion',
        '5': 'put the lemon into the gray plate',
        '6': 'put the onion into the gray plate',
        '7': 'pick up the apple',
        '8': 'pick up the lemon',
        '9': 'pick up the mango',
        '10': 'put the tomato into the pink plate',
        '11': 'pick up the bitter melon',
        '12': 'put the mango into the pink plate',
        '13': 'pick up the peach',
        '14': 'put the peach into the bamboo basket',
        '15': 'pick up the pear',
        '16': 'pick up the tomato',
        '17': 'put the bitter melon into the pink plate',
        '18': 'pull up to pull out the tape on the gray base',
        '19': 'clamp the torn tape on the gray base',
        '20': 'press down to cut the tape on the gray base',
        '21': 'put the pear into the bamboo basket'
    },
    dataset=dict(
        type='PrivateInferenceDataset',
        embodiment_id=31,
        img_keys=['cam_high', 'cam_left_wrist'],
        transforms=[
            dict(
                type='ProcessPromptsWithImage',
                max_len=600,
                num_images=2,
                tokenizer=dict(type='PretrainedTokenizer'
                               # special_tokens={'pad_token': '<PAD>'}
                               )),
            dict(type='ResizeImages', height=224, width=224),
            dict(
                type='NormalizeImages',
                means=[[123.515625, 116.04492188, 103.59375],
                       [123.515625, 116.04492188, 103.59375]],
                stds=[[58.27148438, 57.02636719, 57.27539062],
                      [58.27148438, 57.02636719, 57.27539062]],
            ),
            dict(
                type='NormalizeStatesAndActions',
                state_dim=16,
                state_key='proprio',
                action_key='action',
                norm_type='mean_std')
        ]),
    denormalize_action=dict(
        type='DenormalizePrivateAction',
        norm_type='mean_std',
        action_dim=16,
    ),
    operator=dict(
        type='UROperator',
        img_left_topic='/wrist_camera/color/image_raw',
        img_front_topic='/front_camera/color/image_raw',
        puppet_arm_left_topic='/joint_states',
        puppet_gripper_left_topic='/gripper/position',
        puppet_ee_pose_left_topic='/arm/tcp_pose',
        use_depth_image=False))
