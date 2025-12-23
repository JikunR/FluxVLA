model = dict(
    type='OpenVLA',
    model_path='openvla/openvla-7b-finetuned-libero-10',
    use_quantization=False,
    use_lora=True,
    lora_rank=32,
    lora_dropout=0.0,
    lora_target_modules='all-linear')

train_dataloader = dict(
    per_device_batch_size=8,
    dataset=dict(
        type='DistributedRepeatingDataset',
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action']
        },
        dataset_statistics=dict(
            libero_10_no_noops=dict(
                proprio=dict(
                    mean=[
                        -0.0419132679050224, 0.034591788297521735,
                        0.8265881844959498, 2.90259518190321,
                        -0.5570652600832564, -0.16592166873533284,
                        0.02845031351083622, -0.02880236273799356
                    ],
                    std=[
                        0.03756502182067285, 0.05091765880150317,
                        0.09107525593038836, 0.12327524826514363,
                        0.4418352294043351, 0.12490994022681218,
                        0.004662133639412193, 0.00460807817987938
                    ],
                    min=[
                        -0.48278069496154785, -0.3309336006641388,
                        0.44550687074661255, 1.1323540210723877,
                        -3.6312508583068848, -1.842738389968872,
                        -0.005453015677630901, -0.04112039878964424
                    ],
                    max=[
                        0.2103137969970703, 0.38887521624565125,
                        1.333192229270935, 3.7248642444610596, 3.5618896484375,
                        1.3863215446472168, 0.041575800627470016,
                        0.0013126095291227102
                    ],
                    q01=[
                        -0.1855636807291125, -0.16145669766439186,
                        0.7064185725262808, 2.5678211534702324,
                        -1.2430377303522737, -0.5195810482339626,
                        0.01022917473133343, -0.03999379658232052
                    ],
                    q99=[
                        0.05938728483051665, 0.2361478409238694,
                        0.9397258571145816, 3.2118708728143526,
                        0.49082919816100534, 0.2100883989120329,
                        0.040047131839991014, -0.011104049991952391
                    ]),
                timestamp=dict(
                    mean=[7.007510548523206],
                    std=[4.457129586378845],
                    min=[0.0],
                    max=[25.2],
                    q01=None,
                    q99=None),
                action=dict(
                    mean=[
                        0.01905656634877842, 0.05672475971568838,
                        -0.056239289430234256, 0.004756678478841528,
                        0.002797492338491304, -0.00714607048416358,
                        0.54599156235075
                    ],
                    std=[
                        0.10588348353857541, 0.13552477199270377,
                        0.13886650724555177, 0.01433739270759898,
                        0.02038583948325967, 0.033299202425577934,
                        0.1881810653484855
                    ],
                    min=[
                        -0.9375, -0.9375, -0.9375, -0.23642857372760773,
                        -0.3053571283817291, -0.3642857074737549, 0.0
                    ],
                    max=[
                        0.9375, 0.9375, 0.9375, 0.32892856001853943,
                        0.36964285373687744, 0.375, 1.0
                    ],
                    q01=[
                        -0.4997477764535965, -0.6992653512084763,
                        -0.6543309163615124, -0.07417070079989778,
                        -0.11898748445770971, -0.15976085962510805, 0.0
                    ],
                    q99=[
                        0.658747846713789, 0.7333480638990948,
                        0.768601965587579, 0.09784501244893279,
                        0.12943469061349036, 0.15137893471596325, 1.0
                    ]))),
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name='libero_10_no_noops',
        datasets=dict(
            type='ParquetDataset',
            data_root_path=  # noqa: E251
            '/limx/tos/limx_mani_data/raw_data/LIBERO_lerobot/libero_10_no_noops_1.0.0_lerobot',  # noqa: E501
            transforms=[
                dict(
                    type='ProcessParquetInputs',
                    parquet_keys=[
                        'observation.state', 'timestamp', 'actions', 'info',
                        'stats', 'action_masks'
                    ],
                    video_keys=[
                        'observation.images.image',
                        'observation.images.image',
                    ],
                    name_mappings={
                        'observation.state': ['states'],
                        'actions': ['actions']
                    }),
                dict(
                    type='NormalizeStatesAndActions',
                    action_dim=14,
                    state_key='proprio',
                    action_key='action',
                    norm_type='quantile'),
                dict(
                    type='ParquetPrompter',
                    action_tokenizer=dict(
                        type='ActionTokenizer',
                        model_path=  # noqa: E251
                        'openvla/openvla-7b-finetuned-libero-10',  # noqa: E501
                        bins=256,
                        min_action=-1,
                        max_action=1,
                    )),
                dict(
                    type='ProcessPrompts',
                    max_len=None,
                    tokenizer=dict(
                        type='PretrainedTokenizer',
                        model_path=  # noqa: E251
                        'openvla/openvla-7b-finetuned-libero-10',  # noqa: E501
                        # special_tokens={'pad_token': '<PAD>'}
                    ),
                    with_labels=True),
                dict(type='ResizeImages', height=224, width=224),
                dict(
                    type='NormalizeImages',
                    means=[[123.515625, 116.04492188, 103.59375],
                           [123.515625, 116.04492188, 103.59375]],
                    stds=[[58.27148438, 57.02636719, 57.27539062],
                          [58.27148438, 57.02636719, 57.27539062]],
                ),
            ],
            action_window_size=1,
            action_key='action',
            use_delta=False,
            statistic_name='libero_10_no_noops',
            window_start_idx=0)))

processor = dict(
    type='PretrainedProcessor',
    model_path=  # noqa: E251
    'openvla/openvla-7b-finetuned-libero-10',  # noqa: E501
    trust_remote_code=True)

runner = dict(
    type='DDPHFFinetuneRunner',
    learning_rate=0.0005,
    sampler=None,
    collator=dict(
        type='DictCollator',
        keys=[
            'states', 'observation.eepose', 'timestamp', 'images', 'img_masks',
            'lang_tokens', 'lang_masks', 'actions', 'action_masks', 'labels'
        ],
        meta_keys=['task_description', 'prompt', 'info', 'stats']),
    grad_accumulation_steps=1,
    max_epochs=24,
    save_latest_checkpoint_only=True)

eval = dict(
    type='HFLiberoEvalRunner',
    model_family='openvla',
    processor=dict(type='PretrainedProcessor', trust_remote_code=True),
    task_suite_name='libero_10',
    resize_size=224,
    num_trials_per_task=50,
    num_steps_wait=10,
    seed=7)
