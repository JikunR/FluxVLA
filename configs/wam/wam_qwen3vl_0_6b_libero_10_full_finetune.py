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
#
# Minimal WAM on LIBERO-10 with Qwen3VL-0.6B as the context backbone.

import os

_repo_root = os.path.abspath(os.environ.get('FLUXVLA_ROOT', '.'))
_ckpt_root = os.path.join(_repo_root, 'checkpoints')
_wan_checkpoint_root = os.path.join(_ckpt_root, 'Wan2.2-TI2V-5B')
_action_dit = os.path.join(
    _ckpt_root, 'ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt')
_qwen3vl_model_path = os.path.abspath(
    os.environ.get('QWEN3VL_0_6B_PATH',
                   os.path.join(_ckpt_root, 'LimX-Qwen3-VL-0.6B')))
_qwen3vl_tokenizer_path = os.path.abspath(
    os.environ.get('QWEN3VL_0_6B_TOKENIZER_PATH', _qwen3vl_model_path))
_data_root = os.path.join(_repo_root,
                          'datasets/libero_10_no_noops_lerobotv2.1')

_frame_window_size = 9
_mode_probs = dict(forward=1.0, idm=1.0, policy=1.0, joint=1.0)

model = dict(
    type='WAMVLA',
    pretrained_name_or_path=None,
    num_views=2,
    frame_window_size=_frame_window_size,
    proprio_dim=8,
    action_horizon=32,
    mot_checkpoint_mixed_attn=True,
    freeze_vlm_backbone=False,
    vlm_backbone=dict(
        type='Qwen3VL',
        vlm_backbone_id='qwen3_0.6b_vl_pt',
        vlm_path=_qwen3vl_model_path,
        vlm_config=None,
        use_projection=True,
        projection_output_dim=4096,
        projection_type='linear',
        attn_implementation='sdpa',
        torch_dtype='bf16',
    ),
    video_latent_codec=dict(
        type='Wan22VAE',
        checkpoint_root=_wan_checkpoint_root,
    ),
    vla_head=dict(
        type='WAMHead',
        video_expert=dict(
            type='WanVideoDiT',
            checkpoint_root=_wan_checkpoint_root,
            skip_load_from_pretrain=False,
            config=dict(
                has_image_input=False,
                patch_size=[1, 2, 2],
                in_dim=48,
                hidden_dim=3072,
                ffn_dim=14336,
                freq_dim=256,
                text_dim=4096,
                out_dim=48,
                num_heads=24,
                attn_head_dim=128,
                num_layers=30,
                eps=1.0e-06,
                seperated_timestep=True,
                require_clip_embedding=False,
                require_vae_embedding=False,
                fuse_vae_embedding_in_latents=True,
                video_attention_mask_mode='first_frame_causal',
                action_conditioned=False,
                action_dim=7,
                action_group_causal_mask_mode='group_diagonal',
                use_gradient_checkpointing=True,
            ),
        ),
        action_expert=dict(
            type='ActionDiT',
            pretrained_path=_action_dit,
            skip_load_from_pretrain=False,
            config=dict(
                action_dim=7,
                hidden_dim=1024,
                ffn_dim=4096,
                num_heads=24,
                attn_head_dim=128,
                num_layers=30,
                text_dim=4096,
                freq_dim=256,
                eps=1.0e-06,
                use_gradient_checkpointing=True,
            ),
        ),
        video_scheduler=dict(
            train_shift=5.0, infer_shift=5.0, num_train_timesteps=1000),
        action_scheduler=dict(
            train_shift=5.0, infer_shift=5.0, num_train_timesteps=1000),
        loss=dict(
            lambda_forward_video=1.0,
            lambda_idm_action=1.0,
            lambda_policy_action=1.0,
            lambda_joint_video=1.0,
            lambda_joint_action=1.0,
        ),
        video_cond_noise_prob=0.5,
    ),
)

inference_model = model.copy()

train_dataloader = dict(
    per_device_batch_size=8,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedRepeatingDataset',
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name='libero_10_no_noops',
        datasets=dict(
            type='ParquetDataset',
            data_root_path=_data_root,
            transforms=[
                dict(
                    type='ProcessParquetInputs',
                    parquet_keys=[
                        'observation.state',
                        'timestamp',
                        'actions',
                        'info',
                        'stats',
                        'action_masks',
                    ],
                    video_keys=[
                        'observation.images.image',
                        'observation.images.wrist_image',
                    ],
                    name_mappings={
                        'observation.state': ['states'],
                        'actions': ['actions'],
                    },
                    embodiment_id=0,
                ),
                dict(
                    type='ResizeImages',
                    height=224,
                    width=224,
                ),
                dict(
                    type='NormalizeImages',
                    means=[0.5, 0.5, 0.5],
                    stds=[0.5, 0.5, 0.5],
                    scale_to_unit_interval=True,
                ),
                dict(
                    type='NormalizeStatesAndActions',
                    action_dim=7,
                    state_dim=8,
                    state_key='proprio',
                    action_key='action',
                    norm_type='min_max',
                ),
                dict(
                    type='PrepareVideo',
                    num_views=2,
                    frame_window_size=_frame_window_size,
                    tile_direction='horizontal',
                ),
                dict(
                    type='LiberoPromptFromInputs',
                    tokenizer=dict(
                        type='PretrainedTokenizer',
                        model_path=_qwen3vl_tokenizer_path,
                    ),
                    max_len=128,
                    use_conversation=False,
                ),
            ],
            action_window_size=32,
            action_key='action',
            use_delta=False,
            statistic_name='libero_10_no_noops',
            window_start_idx=0,
            frame_window_size=_frame_window_size,
            frame_sample_stride=4,
        ),
    ),
)

runner = dict(
    type='DDPTrainRunner',
    max_epochs=10,
    optimizer=dict(
        lr=1e-4,
        type='AdamW',
        weight_decay=1e-2,
        betas=(0.9, 0.95),
    ),
    max_grad_norm=1.0,
    collator=dict(
        type='WAMModeCollator',
        mode='batch',
        mode_probs=_mode_probs,
        keys=[
            'states',
            'images',
            'img_masks',
            'actions',
            'action_masks',
            'embodiment_ids',
            'frame_masks',
            'lang_tokens',
            'lang_masks',
            'training_mode',
        ],
        meta_keys=['task_description', 'prompt', 'info', 'stats', 'timestamp'],
    ),
    sampler=None,
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        window_size=1,
    ),
    lr_scheduler=dict(type='linear-warmup+cosine-decay', warmup_ratio=0.05),
    enable_gradient_checkpointing=True,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
)

eval = dict(
    type='LiberoEvalRunner',
    task_suite_name='libero_10',
    model_family='wam',
    norm_stats_key='libero_10_no_noops',
    eval_chunk_size=10,
    resize_size=224,
    num_trials_per_task=50,
    num_steps_wait=30,
    seed=42,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    dataset=dict(
        type='LiberoParquetEvalDataset',
        img_buffer_len=1,
        transforms=[
            dict(
                type='ProcessLiberoEvalInputs',
                img_keys=['agentview_image', 'robot0_eye_in_hand_image'],
            ),
            dict(
                type='TransformImage',
                image_resize_strategy='resize-naive',
                input_sizes=[[3, 224, 224], [3, 224, 224]],
                means=[[127.5, 127.5, 127.5], [127.5, 127.5, 127.5]],
                stds=[[127.5, 127.5, 127.5], [127.5, 127.5, 127.5]],
            ),
            dict(
                type='LiberoProprioFromInputs',
                norm_type='min_max',
                out_key='states',
                stat_key='proprio',
                state_dim=8,
            ),
            dict(
                type='LiberoPromptFromInputs',
                tokenizer=dict(
                    type='PretrainedTokenizer',
                    model_path=_qwen3vl_tokenizer_path,
                ),
                max_len=128,
                use_conversation=False,
            ),
            dict(
                type='PrepareVideo',
                num_views=2,
                frame_window_size=1,
                tile_direction='horizontal',
            ),
        ],
    ),
    denormalize_action=dict(
        type='DenormalizeLiberoAction',
        norm_type='min_max',
        action_dim=7,
    ),
)
