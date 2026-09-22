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
"""Cosmos3-Nano training and evaluation on RoboDojo ARX-X5.

The policy consumes all three RoboDojo cameras and absolute 14D joint
states/actions.  Its action-policy settings follow the official Cosmos3 DROID
recipe in the sibling ``cosmos-framework`` checkout: 32-step chunks (33 video
frames), structured JSON prompts, CFG dropout, 10k updates, and a 5x
learning-rate multiplier for the action projections. State conditioning is
disabled because it regressed RoboDojo closed-loop simulation performance.

Training (16 GPUs; global batch 2048):
    torchrun --nproc-per-node=8 --nnodes=2 scripts/train.py \
        --config \
        configs/cosmos3/cosmos3nano_robodojo_full_finetune.py \
        --work-dir work_dirs/cosmos3nano_robodojo_full_finetune

Evaluation is served through the RoboDojo/FluxThemis manager using the
``eval`` and ``themis`` sections below.
"""

_base_ = './cosmos3nano_libero_10_full_finetune.py'

_ROBODOJO_DATA_ROOT = './datasets/RoboDojo_lerobot_v21_video'
_STATISTIC_NAME = 'robodojo_arx_x5'
_ACTION_DIM = 14
_MAX_ACTION_DIM = 64
_MAX_STATE_DIM = 64
_ACTION_HORIZON = 32
_FRAME_WINDOW_SIZE = _ACTION_HORIZON + 1
_CONDITIONING_FPS = 25.0
_EMBODIMENT_ID = 31
_PREPEND_STATE_TO_ACTION = False

# Preserve RoboDojo's native 4:3 camera aspect ratio while downsampling each
# 640x480 view. The three views are arranged as one overhead view above two
# half-height wrist views, then padded to Cosmos3's official 256-tier 3:4
# video bucket: VIDEO_RES_SIZE_INFO['256']['3,4'] == (W=256, H=320).
_IMAGE_HEIGHT = 192
_IMAGE_WIDTH = 256
_VIDEO_HEIGHT = 320
_VIDEO_WIDTH = 256

_ROBODOJO_GENERALIZATION_BASE_TASKS = (
    'stack_bowls',
    'push_T',
    'pack_objects_into_box',
    'fold_clothes',
    'hang_mugs',
    'sweep_blocks',
    'pour_liquid_into_cup',
    'make_toast',
    'arrange_largest_number',
    'sort_nesting_dolls_by_size',
    'store_laptop_and_headphones',
    'stack_blocks',
)
_ROBODOJO_EPISODE_OVERRIDES = {
    task_name: 25
    for base_task in _ROBODOJO_GENERALIZATION_BASE_TASKS
    for task_name in (base_task, f'{base_task}_random')
}

_ROBODOJO_TRANSFORMS = [
    dict(
        type='ProcessParquetInputs',
        embodiment_id=_EMBODIMENT_ID,
        parquet_keys=[
            'observation.state',
            'timestamp',
            'actions',
            'info',
            'stats',
            'action_masks',
        ],
        video_keys=[
            'observation.images.cam_high',
            'observation.images.cam_left_wrist',
            'observation.images.cam_right_wrist',
        ],
        name_mappings={
            'observation.state': ['states'],
            'actions': ['actions'],
        }),
    dict(type='ResizeImages', height=_IMAGE_HEIGHT, width=_IMAGE_WIDTH),
    dict(
        type='AugVideo',
        rotation_range=0.0,
        brightness_range=(0.7, 1.3),
        contrast_range=(0.6, 1.4),
        crop_scale=(0.95, 0.95),
        crop_ratio=(1.0, 1.0),
        prob=1.0,
        saturation_range=(0.5, 1.5),
        hue_delta=0.08),
    dict(
        type='ProcessCosmos3Prompt',
        tokenizer={{_base_._cosmos3_nano_tokenizer}},
        max_len=512,
        cfg_dropout_rate=0.1,
        format_prompt_as_json=True,
        action_metadata=dict(
            append_viewpoint=False,
            viewpoint='concat_view',
            frame_window_size=_FRAME_WINDOW_SIZE,
            conditioning_fps=_CONDITIONING_FPS,
            video_height=_VIDEO_HEIGHT,
            video_width=_VIDEO_WIDTH)),
    dict(type='SimpleNormalizeImages'),
    dict(
        type='NormalizeStatesAndActions',
        action_dim=_MAX_ACTION_DIM,
        state_dim=_MAX_STATE_DIM,
        state_key='proprio',
        action_key='action',
        norm_type='quantile'),
    dict(
        type='BuildCosmos3Sequence',
        raw_action_dim=_ACTION_DIM,
        mode='wam',
        frame_window_size=_FRAME_WINDOW_SIZE,
        prepend_state_to_action=_PREPEND_STATE_TO_ACTION,
        conditioning_fps=_CONDITIONING_FPS),
    dict(
        type='PrepareVideo',
        num_views=3,
        frame_window_size=_FRAME_WINDOW_SIZE,
        tile_direction='top_bottom_pair',
        top_view=0,
        bottom_views=(1, 2),
        bottom_height_ratio=0.5),
    dict(type='ResizeAndReflectPad', height=_VIDEO_HEIGHT, width=_VIDEO_WIDTH),
]

# Only the task-specific dimensions and fresh action-policy initialization
# differ from the inherited Cosmos3-Nano architecture.
model = dict(
    ori_action_dim=_ACTION_DIM,
    action_horizon=_ACTION_HORIZON,
    reinitialize_action_policy=True,
    vision_vae=dict(encode_exact_durations=[_FRAME_WINDOW_SIZE]),
)

inference_model = dict(
    ori_action_dim=_ACTION_DIM,
    action_horizon=_ACTION_HORIZON,
    reinitialize_action_policy=True,
    vision_vae=dict(
        encode_exact_durations=[_FRAME_WINDOW_SIZE],
        pretrained_name_or_path=None),
)

train_dataloader = dict(
    _delete_=True,
    per_device_batch_size=16,
    per_device_num_workers=4,
    prefetch_factor=1,
    dataset=dict(
        type='DistributedRepeatingDataset',
        seed=42,
        reshuffle_each_epoch=True,
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'action', 'timestamp'],
        statistic_name=_STATISTIC_NAME,
        datasets=dict(
            type='ParquetDataset',
            data_root_path=[_ROBODOJO_DATA_ROOT],
            transforms=_ROBODOJO_TRANSFORMS,
            action_window_size=_ACTION_HORIZON,
            action_key='action',
            use_delta=False,
            statistic_name=_STATISTIC_NAME,
            window_start_idx=0,
            frame_window_size=_FRAME_WINDOW_SIZE,
            require_full_window=True)),
)

# Match the official Cosmos3 LIBERO optimization scale: global batch 2048,
# base LR 5e-5, and 5x LR for the action projections. With 16 GPUs this is
# 16 samples/GPU * 16 GPUs * 8 accumulation steps = 2048.
runner = dict(
    max_steps=10000,
    save_iter_interval=1000,
    max_keep_ckpts=2,
    grad_accumulation_steps=8,
    optimizer=dict(
        lr=5e-5,
        paramwise_learning_rate={
            'action_in_proj.': 2.5e-4,
            'action_out_proj.': 2.5e-4,
            'action_modality_embed': 2.5e-4,
        }),
    lr_scheduler=dict(
        type='linear-warmup+linear-decay', warmup_steps=0,
        cycle_length=100000),
    metric=dict(grad_accumulation_steps=8),
)

eval = dict(
    _delete_=True,
    report_kind='robodojo',
    task_suite_name='robodojo',
    model_family='cosmos3',
    eval_chunk_size=_ACTION_HORIZON,
    num_trials_per_task=50,
    num_trials_per_task_overrides=_ROBODOJO_EPISODE_OVERRIDES,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    dataset=dict(
        type='RoboDojoEvalDataset',
        unnorm_key=_STATISTIC_NAME,
        transforms=[
            dict(
                type='ProcessEvalInputs',
                img_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
                embodiment_id=_EMBODIMENT_ID),
            dict(
                type='StateFromInputs',
                stat_key='proprio',
                norm_type='quantile',
                state_dim=_MAX_STATE_DIM),
            dict(
                type='SetCosmos3ActionMetadata',
                conditioning_fps=_CONDITIONING_FPS,
                prepend_state_to_action=_PREPEND_STATE_TO_ACTION),
            dict(
                type='ProcessCosmos3Prompt',
                tokenizer={{_base_._cosmos3_nano_tokenizer}},
                max_len=512,
                cfg_dropout_rate=0.0,
                format_prompt_as_json=True,
                action_metadata=dict(
                    append_viewpoint=False,
                    viewpoint='concat_view',
                    frame_window_size=_FRAME_WINDOW_SIZE,
                    conditioning_fps=_CONDITIONING_FPS,
                    video_height=_VIDEO_HEIGHT,
                    video_width=_VIDEO_WIDTH),
                output_key='lang_tokens',
                output_attention_mask_key='lang_masks'),
            dict(
                type='TransformImage',
                image_resize_strategy='resize-naive',
                input_sizes=[[3, _IMAGE_WIDTH, _IMAGE_HEIGHT]] * 3,
                means=[[127.5, 127.5, 127.5]] * 3,
                stds=[[127.5, 127.5, 127.5]] * 3),
            dict(
                type='PrepareVideo',
                num_views=3,
                frame_window_size=1,
                tile_direction='top_bottom_pair',
                top_view=0,
                bottom_views=(1, 2),
                bottom_height_ratio=0.5),
            dict(
                type='ResizeAndReflectPad',
                height=_VIDEO_HEIGHT,
                width=_VIDEO_WIDTH),
        ]),
    denormalize_action=dict(
        type='DenormalizePrivateAction',
        norm_type='quantile',
        action_dim=_ACTION_DIM,
        statistic_name=_STATISTIC_NAME),
)

themis = dict(
    transport=dict(
        host='127.0.0.1',
        port=5555,
        timeout_s=30.0,
        image_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        state_keys=['states'],
        unnorm_key=_STATISTIC_NAME,
        image_encoding='rgb8',
        report_service_name='/fluxvla/report_evaluation'),
    runner=dict(
        type='EvalRunner',
        environment=dict(
            type='RoboDojoEnvironment',
            task_name='all',
            env_cfg_type='arx_x5',
            robodojo_root='/root/projects/RoboDojo',
            device_id=1,
            action_mode='joint',
            headless=True,
            save_videos=False),
        model_client=dict(type='FluxVLAZMQModelClient'),
        evaluator=dict(type='SuccessRateEvaluator'),
        seed=0,
        episodes_per_task=50,
        episodes_per_task_overrides=_ROBODOJO_EPISODE_OVERRIDES,
        max_episode_steps=2000,
        execute_horizon=_ACTION_HORIZON,
        stop_on_success=True,
        parallel_workers=1,
        simulator_gpu_ids=None,
        work_dir='work_dirs/fluxthemis'),
    ros_server=dict(
        dataset_section='eval',
        evaluation_reporting=dict(
            result_output_dir='work_dirs/fluxthemis', report_kind='robodojo'),
        device='cuda:0',
        workers=dict(
            startup_timeout_s=900.0,
            request_timeout_s=120.0,
            lease_timeout_s=900.0),
        mixed_precision_dtype='bf16',
        enable_mixed_precision=True,
        model_outputs_environment_actions=False,
        forward_seed=False,
        denormalize_context=dict(task_suite_name=_STATISTIC_NAME)),
)
