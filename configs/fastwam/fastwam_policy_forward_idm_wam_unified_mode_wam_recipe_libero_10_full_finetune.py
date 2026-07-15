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
# Ablation: FastWAM WAM-style unified mode with WAM-baseline action
# padding and optimizer recipe.

_base_ = ('./fastwam_policy_forward_idm_wam_unified_mode_'
          'libero_10_full_finetune.py')

_frame_window_size = 9

train_dataloader = dict(
    dataset=dict(
        datasets=dict(
            type='ParquetDataset',
            data_root_path=  # noqa: E251
            '/mnt/data/cpfs/mnt/data/yanis/FastWAM/data/libero_mujoco3.3.2/libero_10_no_noops_lerobot',  # noqa: E501
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
                    backend='torchvision',
                    scale_to_unit_interval=True,
                ),
                dict(
                    type='NormalizeImages',
                    means=[0.5, 0.5, 0.5],
                    stds=[0.5, 0.5, 0.5],
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
                    type='LoadCachedTextEmbedding',
                    cache_dir=('/mnt/data/cpfs/mnt/data/yanis/FastWAM/data/'
                               'text_embeds_cache/libero'),
                    context_len=128,
                    enc_id='wan22ti2v5b',
                ),
            ],
            action_window_size=32,
            action_key='action',
            use_delta=False,
            statistic_name='libero_10_no_noops',
            window_start_idx=0,
            frame_window_size=_frame_window_size,
            frame_sample_stride=4,
        ), ), )

runner = dict(
    lr_scheduler=dict(
        _delete_=True,
        type='linear-warmup+cosine-decay',
        warmup_ratio=0.05,
        betas=(0.9, 0.95),
    ), )
