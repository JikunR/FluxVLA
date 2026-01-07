model = dict(
    type='PI0FlowMatching',
    llm_backbone=dict(
        type='ConditionGemmaModel',
        adarms_cond_dim=None,
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=2,
        eos_token_id=1,
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
        num_key_value_heads=1,
        rms_norm_eps=1e-06,
        rope_theta=10000.0,
        torch_dtype='float32',
        use_cache=True,
        vocab_size=257152,
    ),
    vision_backbone=dict(
        type='SigLIPViTBackbone',
        vision_backbone_id='siglip_224',
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
            patch_size=14,
            projection_dim=2048,
            projector_hidden_act='gelu_fast',
            torch_dtype='float32',
            vision_use_head=False,
        ),
    ),
    projector=dict(
        type='LinearProjector',
        in_dim=1152,
        out_dim=2048,
    ),
    proj_width=1024,
    n_action_steps=50,
    state_proj=dict(type='LinearProjector', in_dim=32, out_dim=1024),
    action_in_proj=dict(type='LinearProjector', in_dim=32, out_dim=1024),
    action_out_proj=dict(type='LinearProjector', in_dim=1024, out_dim=32),
    action_time_mlp_in=dict(type='LinearProjector', in_dim=2048, out_dim=1024),
    action_time_mlp_out=dict(
        type='LinearProjector', in_dim=1024, out_dim=1024),
    max_action_dim=32,
    llm_expert=dict(
        type='ConditionGemmaModel',
        attention_bias=False,
        adarms_cond_dim=None,
        attention_dropout=0.0,
        bos_token_id=2,
        eos_token_id=1,
        head_dim=256,
        hidden_act='gelu_pytorch_tanh',
        hidden_activation='gelu_pytorch_tanh',
        hidden_size=1024,
        initializer_range=0.02,
        intermediate_size=4096,
        max_position_embeddings=8192,
        model_type='gemma',
        num_attention_heads=8,
        num_hidden_layers=18,
        num_key_value_heads=1,
        pad_token_id=0,
        rms_norm_eps=1e-06,
        rope_theta=10000.0,
        torch_dtype='float32',
        transformers_version='4.48.1',
        use_adarms=False,
        use_cache=True,
        vocab_size=257152),
    freeze_llm_backbone=False,
    freeze_vision_backbone=False,
    pretrained_name_or_path=  # noqa: E251
    '/limx/tos/users/liyinhao/checkpoints/pi0_base/model.safetensors',  # noqa: E501
    name_mapping={
        'llm_backbone': 'paligemma_with_expert.paligemma.model.language_model',
        'vision_backbone.vision':
        'paligemma_with_expert.paligemma.model.vision_tower',
        'projector.projector':
        'paligemma_with_expert.paligemma.model.multi_modal_projector.linear',
        'llm_expert': 'paligemma_with_expert.gemma_expert.model',
        'action_time_mlp_in.projector': 'action_time_mlp_in',
        'action_time_mlp_out.projector': 'action_time_mlp_out',
        'state_proj.projector': 'state_proj',
        'action_in_proj.projector': 'action_in_proj',
        'action_out_proj.projector': 'action_out_proj',
        'llm_backbone.embed_tokens': 'paligemma_with_expert.paligemma.lm_head',
    },
    params_to_change_dtype=[
        'llm_expert.llm.model.layers',
        'vlm_backbone.vlm.model.language_model.layers',
        'vlm_backbone.vlm.model.vision_tower',
        'vlm_backbone.vlm.model.multi_modal_projector',
    ],
    ori_action_dim=7,
)

inference_model = model.copy()

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
                            '/limx/tos/limx_mani_checkpoints/open_source/huggingface/paligemma-3b-pt-224',  # noqa: E501
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
                        action_dim=32,
                        state_dim=32,
                        state_key='proprio',
                        action_key='action',
                        norm_type='proprio')
                ],
                action_window_size=50)
        ]))

runner = dict(
    type='FSDPTrainRunner',
    stage='vla-full-train',
    max_epochs=3,
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
        '/limx/tos/limx_mani_checkpoints/open_source/huggingface/paligemma-3b-pt-224',  # noqa: E501
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
    lr_scheduler_type='linear-warmup+cosine-decay',
    warmup_ratio=0.1,
    enable_gradient_checkpointing=False,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
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
        img_keys=['cam_high', 'cam_left_wrist'],
        transforms=[
            dict(
                type='ParquetPrompter',
                use_conversation=False,
                add_new_line=True),
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
                state_dim=32,
                state_key='proprio',
                action_key='action',
                norm_type='proprio')
        ]),
    denormalize_action=dict(
        type='DenormalizePrivateAction',
        norm_type='proprio',
    ),
    operator=dict(
        type='UROperator',
        img_left_topic='/wrist_camera/color/image_raw',
        img_front_topic='/front_camera/color/image_raw',
        puppet_arm_left_topic='/joint_states',
        puppet_gripper_left_topic='/gripper/position',
        puppet_ee_pose_left_topic='/arm/tcp_pose',
        use_depth_image=False))
