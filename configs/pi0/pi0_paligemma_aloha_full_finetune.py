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
        datasets=[
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
    type='AlohaInferenceRunner',
    task_descriptions={
        '1':
        'pick up the yellow chicken toy with left arm',
        '2':
        'place it in the brown flat cardboard box with right arm',
        '3':
        'pick up the pruple caterpillar toy with right arm',
        '4':
        'place it in the brown flat cardboard box with left arm',
        '5':
        'pick up the white racing car toy with right arm',
        '6':
        'pick up the white racing car toy with left arm',
        '7':
        'pick up the banana with left arm',
        '8':
        'place it on the blue plate with left arm',
        '9':
        'pick up the small tomato with left arm',
        '10':
        'pick up the lime with right arm',
        '11':
        'place it on the blue plate with right arm',
        '12':
        'hold the blue tissue package with left arm',
        '13':
        'grasp the tissue with right arm',
        '14':
        'place the tissue on the green plate with right arm',
        '15':
        'place the tissue in the gray trash can with left arm',
        '16':
        'pick up the tissue on the green plate with left arm',
        '17':
        'pull the tissue out of the blue package with right arm',
        '18':
        ('grasp the long green stems of off-white flower bouquet on the table '
         'with right arm'),
        '19':
        'place the lid of the black bottle on the table with right arm',
        '20':
        ('return the bottle to its upright position and places it back on the '
         'table with left arm'),
        '21':
        'lift up the black bottle with left arm',
        '22':
        'grasp the lid of the black bottle with right arm',
        '23':
        'place the green bowl on the table with right arm',
        '24':
        'grasp the black bottle with left arm',
        '25': (
            'place the lid of the black bottle on the brown drawer with right arm'  # noqa: E501
        ),
        '26': (
            'lift the bottle near the side of the green bowl that is farther away '  # noqa: E501
            'from the camera with left arm'),
        '27': (
            'pull outward to open the second dark brown drawer from the bottom '  # noqa: E501
            'with left arm'),
        '28':
        'pick up the blue bottle with left arm',
        '29':
        'pick up the small black yellow box with left arm',
        '30': (
            'place it in the second dark brown drawer from the bottom with left arm'  # noqa: E501
        ),
        '31':
        ('grasp the handle of the second dark brown drawer from the bottom '
         'with left arm'),
        '32': (
            'touch the outer surface of the opened dark brown drawer with left arm'  # noqa: E501
        ),
        '33': (
            'rotate the tape dispenser downward to engage the cutting blade and '  # noqa: E501
            'sever the tape with the right arm'),
        '34': (
            'hold down the tape end to ensure it adheres firmly to the box surface '  # noqa: E501
            'with left arm'),
        '35':
        'pull the tape dispenser to the right to seal the box with right arm',
        '36':
        'grasp the yellow silicone lid with left arm',
        '37':
        'grasp the blue silicone lid with right arm',
        '38':
        ('push inward to close the opened dark brown drawer with left arm'),
        '39':
        'pick up the lime with left arm',
        '40':
        'pick up the banana with right arm',
        '41':
        'pick up the green and white glue stick with left arm',
        '42':
        ('cap the dark green cup with the yellow silicone lid with left arm'),
        '43':
        'grasp the pink silicone lid with right arm',
        '44':
        'grasp the yellow silicone lid with right arm',
        '45':
        ('cap the dark green cup with the yellow silicone lid with right arm'),
        '46':
        'grasp the pink silicone lid with left arm',
        '47':
        ('tilt the black bottle to pour its contents into the green bowl '
         'with left arm'),
        '48':
        'grasp the green bowl with right arm',
        '49':
        'pick up the mushroom with right arm',
        '50':
        'grasp the blue silicone lid with left arm',
        '51':
        'pick up the yellow chicken toy with right arm',
        '52':
        'grasp the red bowl with right arm',
        '53':
        'lift up the red bowl with right arm',
        '54':
        ('lift the bottle near the side of the red bowl that is farther away '
         'from the camera with left arm'),
        '55':
        'place the red bowl on the table with right arm',
        '56':
        'tilt the black bottle to pour its contents into the red bowl with left arm',  # noqa: E501
        '57':
        'pick up the pruple caterpillar toy with left arm',
        '58':
        'lift up the green bowl with right arm',
        '59':
        'pick up the corn with right arm',
        '60':
        'place it on the pink plate with left arm',
        '61':
        'pick up the kiwi with left arm',
        '62':
        'place it on the pink plate with right arm',
        '63':
        'pick up the small tomato with right arm',
        '64':
        'pick up the corn with left arm',
        '65':
        'pick up the mushroom with left arm',
        '66':
        'cap the brown cup with the pink silicone lid with right arm',
        '67':
        'pick up the kiwi with right arm',
        '68':
        'place it on the green plate with left arm',
        '69':
        'cap the brown cup with the pink silicone lid with left arm',
        '70':
        'place it on the green plate with right arm',
    },
    seed=7,
    dataset=dict(
        type='PrivateInferenceDataset',
        img_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
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
                       [123.515625, 116.04492188, 103.59375],
                       [123.515625, 116.04492188, 103.59375]],
                stds=[[58.27148438, 57.02636719, 57.27539062],
                      [58.27148438, 57.02636719, 57.27539062],
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
        action_dim=14,
    ),
    action_chunk=50,
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
