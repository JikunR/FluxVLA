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
    ori_action_dim=14,
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
                            type='ParquetPrompter',
                            use_conversation=False,
                            add_new_line=True),
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
                                   [123.515625, 116.04492188, 103.59375],
                                   [123.515625, 116.04492188, 103.59375]],
                            stds=[[58.27148438, 57.02636719, 57.27539062],
                                  [58.27148438, 57.02636719, 57.27539062],
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
                            type='ParquetPrompter',
                            use_conversation=False,
                            add_new_line=True),
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
                                   [123.515625, 116.04492188, 103.59375],
                                   [123.515625, 116.04492188, 103.59375]],
                            stds=[[58.27148438, 57.02636719, 57.27539062],
                                  [58.27148438, 57.02636719, 57.27539062],
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
            ],
            ur3=[
                dict(
                    type='ParquetDataset',
                    data_root_path=  # noqa: E251
                    [
                        '/limx/tos/limx_mani_data/raw_data/RealRobot_UR3_lerobot/20250512_20250516_01',  # noqa: E501
                        '/limx/tos/limx_mani_data/raw_data/RealRobot_UR3_lerobot/20250517_20250524_01',  # noqa: E501
                        '/limx/tos/limx_mani_data/raw_data/RealRobot_UR3_lerobot/20250525_20250531_01',  # noqa: E501
                        '/limx/tos/limx_mani_data/raw_data/RealRobot_UR3_lerobot/20250601_20250605_01',  # noqa: E501
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
                                'observation.images.cam_wrist',
                            ],
                            name_mappings={
                                'observation.state': ['states'],
                                'actions': ['actions']
                            }),
                        dict(
                            type='ParquetPrompter',
                            use_conversation=False,
                            add_new_line=True),
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
                                   [123.515625, 116.04492188, 103.59375],
                                   [123.515625, 116.04492188, 103.59375]],
                            stds=[[58.27148438, 57.02636719, 57.27539062],
                                  [58.27148438, 57.02636719, 57.27539062],
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
            ])))

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
    change_key_name=False)
