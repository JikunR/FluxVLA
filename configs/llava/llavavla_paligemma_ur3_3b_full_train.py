model = dict(
    type='LlavaVLA',
    pretrained_name_or_path=  # noqa: E251
    '/limx/tos/users/liyinhao/cache/openpi/openpi-assets/checkpoints/pi0_libero_pytorch/model.safetensors',  # noqa: E501. Refer to https://github.com/ZibinDong/openpi_pytorch
    vlm_backbone=dict(
        type='PaliGemma',
        vlm_backbone_id='paligemma_3b_pt_224',
        use_llm=True,
        vlm_config=dict(
            vocab_size=257152,
            bos_token_id=2,
            eos_token_id=1,
            hidden_size=2048,
            image_token_index=257152,
            model_type='paligemma',
            pad_token_id=0,
            projection_dim=2048,
            text_config=dict(
                attention_bias=False,
                attention_dropout=0.0,
                head_dim=256,
                hidden_act='gelu_pytorch_tanh',
                hidden_activation='gelu_pytorch_tanh',
                hidden_size=2048,
                initializer_range=0.02,
                intermediate_size=16384,
                max_position_embeddings=8192,
                model_type='gemma',
                num_attention_heads=8,
                num_hidden_layers=18,
                num_image_tokens=256,
                num_key_value_heads=1,
                rms_norm_eps=1e-06,
                rope_theta=10000.0,
                torch_dtype='float32',
                use_cache=True,
                vocab_size=257152,
            ),
            transformers_version='4.52.4',
            vision_config=dict(
                attention_dropout=0.0,
                hidden_act='gelu_pytorch_tanh',
                hidden_size=1152,
                image_size=224,
                intermediate_size=4304,
                layer_norm_eps=1e-06,
                model_type='siglip_vision_model',
                num_attention_heads=16,
                num_channels=3,
                num_hidden_layers=27,
                num_image_tokens=256,
                patch_size=14,
                projection_dim=2048,
                projector_hidden_act='gelu_fast',
                torch_dtype='float32',
                vision_use_head=False,
            ))),
    vla_head=dict(
        type='LlavaActionHead',
        state_dim=7,
        hidden_size=2048,
        num_layers=1,
        num_heads=4,
        traj_length=10,
        action_dim=7),
    freeze_vlm_backbone=False,
    freeze_projector=False,
    name_mapping={
        'vlm_backbone.vlm.model.language_model':
        'model.paligemma_with_expert.paligemma.model.language_model',
        'vlm_backbone.vlm.model.vision_tower':
        'model.paligemma_with_expert.paligemma.model.vision_tower',
        'vlm_backbone.vlm.model.multi_modal_projector':
        'model.paligemma_with_expert.paligemma.model.multi_modal_projector',
    })

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
                        type='ProcessPrompts',
                        tokenizer=dict(
                            type='PretrainedTokenizer',
                            model_path=  # noqa: E251
                            '/limx/tos/limx_mani_checkpoints/open_source/huggingface/pi0',  # noqa: E501
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
                        state_dim=7,
                        action_dim=7,
                        state_key='proprio',
                        action_key='action',
                        norm_type='mean_std')
                ],
                action_window_size=10)
        ]))

runner = dict(
    type='FSDPTrainRunner',
    stage='vla-full-train',
    max_epochs=5,
    learning_rate=2e-5,
    weight_decay=0.0,
    max_grad_norm=1.0,
    collator=dict(
        type='DictCollator',
        keys=[
            'states', 'observation.eepose', 'timestamp', 'images', 'img_masks',
            'lang_tokens', 'lang_masks', 'actions', 'action_masks'
        ],
        meta_keys=['task_description', 'prompt', 'info', 'stats']),
    sampler=None,
    tokenizer=dict(
        type='PretrainedTokenizer',
        model_path=  # noqa: E251
        '/limx/tos/limx_mani_checkpoints/open_source/huggingface/pi0',  # noqa: E501
        # special_tokens={'pad_token': '<PAD>'}
    ),
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
    action_chunk=10,
    dataset=dict(
        type='PrivateInferenceDataset',
        img_keys=['cam_high', 'cam_left_wrist'],
        transforms=[
            dict(
                type='ProcessPrompts',
                tokenizer=dict(type='PretrainedTokenizer')),
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
                state_dim=7,
                state_key='proprio',
                action_key='action',
                norm_type='mean_std')
        ]),
    denormalize_action=dict(
        type='DenormalizePrivateAction',
        norm_type='mean_std',
    ),
    operator=dict(
        type='UROperator',
        img_left_topic='/wrist_camera/color/image_raw',
        img_front_topic='/front_camera/color/image_raw',
        puppet_arm_left_topic='/joint_states',
        puppet_gripper_left_topic='/gripper/position',
        puppet_ee_pose_left_topic='/arm/tcp_pose',
        use_depth_image=False))
