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
    './checkpoints/GR00T-N1.5-3B',
    vlm_backbone=dict(
        type='EagleBackbone',
        vlm_path=  # noqa: E251
        'fluxvla/models/third_party_models/eagle2_hg_model',
        vlm_config=dict(max_input_seq_len=600)),
    vla_head=dict(
        type='FlowMatchingHead',
        state_dim=64,
        hidden_size=1024,
        input_embedding_dim=1536,
        num_layers=1,
        num_heads=4,
        num_inference_timesteps=4,
        traj_length=50,
        action_dim=64,
        ori_action_dim=41),
    freeze_vlm_backbone=False,
    name_mapping={
        'vlm_backbone.vlm': 'backbone.eagle_model',
        'vla_head': 'action_head'
    },
    freeze_projector=False)

# Inference uses the same model (no acceleration optimizations)
inference_model = dict(
    type='LlavaVLA',
    pretrained_name_or_path=  # noqa: E251
    './checkpoints/GR00T-N1.5-3B',
    vlm_backbone=dict(
        type='EagleBackbone',
        vlm_path=  # noqa: E251
        'fluxvla/models/third_party_models/eagle2_hg_model',
        vlm_config=dict(max_input_seq_len=600)),
    vla_head=dict(
        type='FlowMatchingHead',
        state_dim=64,
        hidden_size=1024,
        input_embedding_dim=1536,
        num_layers=1,
        num_heads=4,
        num_inference_timesteps=4,
        traj_length=10,
        action_dim=64,
        ori_action_dim=41),
    freeze_vlm_backbone=False,
    name_mapping={
        'vlm_backbone.vlm': 'backbone.eagle_model',
        'vla_head': 'action_head'
    },
    freeze_projector=False)

train_dataloader = dict(
    per_device_batch_size=32,
    per_device_num_workers=8,
    dataset=dict(
        type='DistributedRepeatingDataset',
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action']
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name='hud04_water',
        datasets=dict(
            type='ParquetDataset',
            data_root_path=  # noqa: E251
            '/home/jace/dataset/real/loco-mani/wbt_water/loco_mani_wbt_water_0404_relative',  # noqa: E501
            transforms=[
                dict(
                    type='ProcessParquetInputs',
                    parquet_keys=[
                        'observation.state', 'timestamp', 'actions', 'info',
                        'stats', 'action_masks'
                    ],
                    video_keys=[
                        'observation.images.head',
                    ],
                    num_padding_imgs=1,
                    embodiment_id=0,
                    name_mappings={
                        'observation.state': ['states']
                    }),
                dict(type='ParquetPrompter'),
                dict(
                    type='ProcessPromptsWithImage',
                    max_len=600,
                    num_images=2,
                    tokenizer=dict(
                        type='PretrainedTokenizer',
                        model_path=  # noqa: E251
                        'fluxvla/models/third_party_models/eagle2_hg_model',
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
                    state_dim=64,
                    action_dim=64,
                    state_key='proprio',
                    action_key='action',
                    norm_type='min_max')
            ],
            action_window_size=50,
            action_key='action',
            use_delta=False,
            statistic_name='hud04_water',
            window_start_idx=0)))

runner = dict(
    type='FSDPTrainRunner',
    max_epochs=90,
    learning_rate=2e-5,
    weight_decay=0.0,
    max_grad_norm=1.0,
    sampler=None,
    tokenizer=dict(
        type='PretrainedTokenizer',
        model_path=  # noqa: E251
        'fluxvla/models/third_party_models/eagle2_hg_model',
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
    lr_scheduler_type='linear-warmup+cosine-decay',
    warmup_ratio=0.03,
    enable_gradient_checkpointing=False,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    change_key_name=False)

inference = dict(
    type='Teleop02InferenceRunner',
    seed=7,
    task_descriptions={
        '1': 'Walk to the table ahead, pick up the water bottle with the left hand and place it into the basket on the right side.',
    },
    mixed_precision_dtype='bf16',
    state_dim=33,
    action_chunk=10,
    publish_rate=30,
    camera_names=['head'],
    dataset=dict(
        type='PrivateInferenceDataset',
        embodiment_id=0,
        img_keys=['head'],
        num_padding_imgs=1,
        transforms=[
            dict(
                type='ProcessPromptsWithImage',
                max_len=600,
                num_images=2,
                tokenizer=dict(
                    type='PretrainedTokenizer',
                    model_path=  # noqa: E251
                    'fluxvla/models/third_party_models/eagle2_hg_model',
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
                state_dim=64,
                state_key='proprio',
                action_key='action',
                norm_type='min_max')
        ]),
    denormalize_action=dict(
        type='DenormalizePrivateAction',
        norm_type='min_max',
        action_dim=41),
    operator=dict(
        type='Teleop02Operator',
        head_rgb_topic='/head/color/image_raw/compressed',
        joint_state_topic='/joint/state',
        finger_state_topic='/brainco1/hand/state',
        finger_cmd_topic='/brainco1/hand/cmd',
        teleop_cmd_topic='/teleop_cmd',
        cmd_vel_topic='/sdk_cmd_vel',
    ))
