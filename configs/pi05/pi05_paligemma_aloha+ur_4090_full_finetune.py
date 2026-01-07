model = dict(
    type='PI05FlowMatching',
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
    action_in_proj=dict(type='LinearProjector', in_dim=32, out_dim=1024),
    action_out_proj=dict(type='LinearProjector', in_dim=1024, out_dim=32),
    time_mlp_in=dict(type='LinearProjector', in_dim=1024, out_dim=1024),
    time_mlp_out=dict(type='LinearProjector', in_dim=1024, out_dim=1024),
    max_action_dim=32,
    llm_expert=dict(
        type='ConditionGemmaModel',
        attention_bias=False,
        adarms_cond_dim=1024,
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
        use_adarms=True,
        use_cache=True,
        vocab_size=257152),
    freeze_llm_backbone=False,
    freeze_vision_backbone=False,
    pretrained_name_or_path=  # noqa: E251
    '/limx/tos/users/liyinhao/checkpoints/pi05_base/model.pth',  # noqa: E501
    name_mapping={
        'llm_backbone': 'paligemma_with_expert.paligemma.model.language_model',
        'vision_backbone.vision':
        'paligemma_with_expert.paligemma.model.vision_tower',
        'projector.projector':
        'paligemma_with_expert.paligemma.model.multi_modal_projector.linear',
        'llm_expert': 'paligemma_with_expert.gemma_expert.model',
        'time_mlp_in.projector': 'time_mlp_in',
        'time_mlp_out.projector': 'time_mlp_out',
        'action_in_proj.projector': 'action_in_proj',
        'action_out_proj.projector': 'action_out_proj',
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
        datasets=dict(
            aloha_4090=[
                dict(
                    type='ParquetDataset',
                    data_root_path=  # noqa: E251
                    [
                        '/limx/tos/limx_mani_data/raw_data/RealRobot_AgileX_aloha_lerobot_v2/20250601_20250615_02_4090',  # noqa: E501
                        '/limx/tos/limx_mani_data/raw_data/RealRobot_AgileX_aloha_lerobot_v2/20250616_20250630_02_4090',  # noqa: E501
                        '/limx/tos/limx_mani_data/raw_data/RealRobot_AgileX_aloha_lerobot_v2/20250701_20250715_02_4090',  # noqa: E501
                        '/limx/tos/limx_mani_data/raw_data/RealRobot_AgileX_aloha_lerobot_v2/20250716_20250731_02_4090',  # noqa: E501
                        '/limx/tos/limx_mani_data/raw_data/RealRobot_AgileX_aloha_lerobot_v2/20250801_20250815_02_4090',  # noqa: E501
                        '/limx/tos/limx_mani_data/raw_data/RealRobot_AgileX_aloha_lerobot_v2/20250816_20250831_02_4090',  # noqa: E501
                        '/limx/tos/limx_mani_data/raw_data/RealRobot_AgileX_aloha_lerobot_v2/20250901_20250930_02_4090',  # noqa: E501
                    ],
                    transforms=[
                        dict(
                            type='ProcessParquetInputs',
                            parquet_keys=[
                                'observation.state', 'timestamp', 'actions',
                                'info', 'stats', 'action_masks'
                            ],
                            video_keys=[
                                'observation.images.cam_high',
                                'observation.images.cam_left_wrist',
                                'observation.images.cam_right_wrist'
                            ],
                            name_mappings={
                                'observation.state': ['states'],
                                'actions': ['actions']
                            }),
                        dict(
                            type='NormalizeStatesAndActions',
                            action_dim=32,
                            state_dim=32,
                            state_key='proprio',
                            action_key='action',
                            norm_type='min_max'),
                        dict(type='PreparePromptWithState'),
                        dict[str, str | dict[str, str]](
                            type='ProcessPrompts',
                            max_len=200,
                            tokenizer=dict(
                                type='PretrainedTokenizer',
                                model_path=  # noqa: E251
                                '/limx/tos/limx_mani_checkpoints/open_source/huggingface/paligemma-3b-pt-224',  # noqa: E501
                                # special_tokens={'pad_token': '<PAD>'}
                            )),
                        dict(type='ResizeImages', height=224, width=224),
                        dict(type='SimpleNormalizeImages'),
                    ],
                    action_window_size=50)
            ],
            aloha_4060=[
                dict(
                    type='ParquetDataset',
                    data_root_path=  # noqa: E251
                    [
                        '/limx/tos/limx_mani_data/raw_data/RealRobot_AgileX_aloha_lerobot/20250401_20250430_01_4060',  # noqa: E501
                        '/limx/tos/limx_mani_data/raw_data/RealRobot_AgileX_aloha_lerobot/20250501_20250531_01_4060',  # noqa: E501
                        '/limx/tos/limx_mani_data/raw_data/RealRobot_AgileX_aloha_lerobot/20250601_20250701_01_4060',  # noqa: E501
                    ],
                    transforms=[
                        dict(
                            type='ProcessParquetInputs',
                            parquet_keys=[
                                'observation.state', 'timestamp', 'actions',
                                'info', 'stats', 'action_masks'
                            ],
                            video_keys=[
                                'observation.images.cam_high',
                                'observation.images.cam_left_wrist',
                                'observation.images.cam_right_wrist'
                            ],
                            name_mappings={
                                'observation.state': ['states'],
                                'actions': ['actions']
                            }),
                        dict(
                            type='NormalizeStatesAndActions',
                            action_dim=32,
                            state_dim=32,
                            state_key='proprio',
                            action_key='action',
                            norm_type='min_max'),
                        dict(type='PreparePromptWithState'),
                        dict[str, str | dict[str, str]](
                            type='ProcessPrompts',
                            max_len=200,
                            tokenizer=dict(
                                type='PretrainedTokenizer',
                                model_path=  # noqa: E251
                                '/limx/tos/limx_mani_checkpoints/open_source/huggingface/paligemma-3b-pt-224',  # noqa: E501
                                # special_tokens={'pad_token': '<PAD>'}
                            )),
                        dict(type='ResizeImages', height=224, width=224),
                        dict(type='SimpleNormalizeImages'),
                    ],
                    action_window_size=50)
            ],
            ur3=[
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
                                'observation.state', 'timestamp', 'actions',
                                'info', 'stats', 'action_masks'
                            ],
                            video_keys=[
                                'observation.images.cam_high',
                                'observation.images.cam_wrist',
                                'observation.images.cam_wrist'
                            ],
                            name_mappings={
                                'observation.state': ['states'],
                                'actions': ['actions']
                            }),
                        dict(
                            type='NormalizeStatesAndActions',
                            action_dim=32,
                            state_dim=32,
                            state_key='proprio',
                            action_key='action',
                            norm_type='min_max'),
                        dict(type='PreparePromptWithState'),
                        dict[str, str | dict[str, str]](
                            type='ProcessPrompts',
                            max_len=200,
                            tokenizer=dict(
                                type='PretrainedTokenizer',
                                model_path=  # noqa: E251
                                '/limx/tos/limx_mani_checkpoints/open_source/huggingface/paligemma-3b-pt-224',  # noqa: E501
                                # special_tokens={'pad_token': '<PAD>'}
                            )),
                        dict(type='ResizeImages', height=224, width=224),
                        dict(type='SimpleNormalizeImages'),
                    ],
                    action_window_size=50)
            ])))

runner = dict(
    type='FSDPTrainRunner',
    max_epochs=24,
    learning_rate=5e-5,
    weight_decay=0.01,
    max_grad_norm=1.0,
    collator=dict(
        type='DictCollator',
        keys=[
            'states', 'observation.eepose', 'timestamp', 'images', 'img_masks',
            'lang_tokens', 'lang_masks', 'actions', 'action_masks'
        ],
        meta_keys=['task_description', 'prompt', 'info', 'stats']),
    sampler=None,
    warmup_ratio=0.03,
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
    enable_gradient_checkpointing=False,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    change_key_name=False)
