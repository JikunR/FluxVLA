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

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Dict, List

import imageio.v2 as imageio
import numpy as np
import torch
from mmengine import Config
from PIL import Image
from safetensors.torch import load_file

import fluxvla  # noqa: F401
from fluxvla.engines import build_dataset_from_cfg, build_vla_from_cfg
from fluxvla.engines.utils.torch_utils import \
    configure_inference_attention_defaults


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run one-sample open-loop WAM ALOHA visualization.')
    parser.add_argument(
        '--config',
        default='configs/wam/wam_aloha_full_finetune.py',
        help='WAM ALOHA config path.')
    parser.add_argument(
        '--ckpt-path',
        default=('work_dirs/wam_fastwam_aloha_step_020000_remap/checkpoints/'
                 'step_020000_remapped.safetensors'),
        help='Remapped WAM checkpoint.')
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Output directory. Defaults under checkpoint work dir.')
    parser.add_argument(
        '--sample-index',
        type=int,
        default=0,
        help='Global dataset sample index.')
    parser.add_argument(
        '--num-inference-steps',
        type=int,
        default=20,
        help='Flow-matching denoise steps.')
    parser.add_argument(
        '--rollout-chunks',
        type=int,
        default=4,
        help='Autoregressive policy-forward chunks to render.')
    parser.add_argument(
        '--next-frame-index',
        type=int,
        default=2,
        help='Generated frame index reused as next autoregressive input.')
    parser.add_argument(
        '--whole-episode',
        action='store_true',
        help='Roll both teacher-forced and autoregressive videos to the end '
        'of the source episode.')
    parser.add_argument(
        '--ar-only',
        action='store_true',
        help='With --whole-episode, only render the autoregressive video.')
    parser.add_argument(
        '--gt-action-forward-only',
        action='store_true',
        help='With --whole-episode, only render forward dynamics conditioned '
        'on dataset ground-truth actions.')
    parser.add_argument('--fps', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--device',
        default='cuda',
        help='Inference device. Use CUDA_VISIBLE_DEVICES to pick the GPU.')
    return parser.parse_args()


def _default_output_dir(ckpt_path: str, sample_index: int) -> Path:
    work_dir = Path(ckpt_path).resolve().parent.parent
    return work_dir / (
        f'open_loop_aloha_sample_{sample_index:06d}_policy_action_forward_ar')


def _fastwam_field_to_flux(field_stats: Dict) -> Dict:
    return {
        'mean': field_stats['global_mean'],
        'std': field_stats['global_std'],
        'min': field_stats['global_min'],
        'max': field_stats['global_max'],
        'q01': field_stats['global_q01'],
        'q99': field_stats['global_q99'],
    }


def load_dataset_stats(stats_path: Path, statistic_name: str) -> Dict:
    with stats_path.open('r', encoding='utf-8') as f:
        stats = json.load(f)
    if statistic_name in stats:
        dataset_stats = stats[statistic_name]
        if 'proprio' in dataset_stats and 'action' in dataset_stats:
            return {statistic_name: dataset_stats}
    return {
        statistic_name: {
            'proprio': _fastwam_field_to_flux(stats['state']['default']),
            'action': _fastwam_field_to_flux(stats['action']['default']),
        }
    }


def denormalize_action(action: np.ndarray, flux_stats: Dict) -> np.ndarray:
    stats = flux_stats['action']
    mean = np.asarray(stats['mean'], dtype=np.float32)
    std = np.asarray(stats['std'], dtype=np.float32)
    action = np.asarray(action, dtype=np.float32)
    denorm = action.copy()
    dim = min(denorm.shape[-1], mean.shape[0])
    denorm[..., :dim] = denorm[..., :dim] * (std[:dim] + 1e-6) + mean[:dim]
    return denorm


def normalize_state(state: np.ndarray, flux_stats: Dict) -> np.ndarray:
    stats = flux_stats['proprio']
    mean = np.asarray(stats['mean'], dtype=np.float32)
    std = np.asarray(stats['std'], dtype=np.float32)
    state = np.asarray(state, dtype=np.float32)
    norm = np.zeros_like(state, dtype=np.float32)
    dim = min(norm.shape[-1], mean.shape[0])
    norm[..., :dim] = (state[..., :dim] - mean[:dim]) / (std[:dim] + 1e-6)
    return norm


def active_action_dim(action: np.ndarray, flux_stats: Dict) -> int:
    return min(int(action.shape[-1]), len(flux_stats['action']['mean']))


def batch_tensor(value, *, dtype=None, device=None):
    tensor = torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if tensor.ndim in (1, 3, 4):
        tensor = tensor.unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor


def normalized_video_to_frames(video: np.ndarray) -> List[Image.Image]:
    # video: [C, T, H, W] in [-1, 1]
    frames = []
    video = np.asarray(video, dtype=np.float32)
    video = ((video + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    for idx in range(video.shape[1]):
        frame = np.transpose(video[:, idx], (1, 2, 0))
        frames.append(Image.fromarray(frame))
    return frames


def save_video(frames: List[Image.Image], path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = [np.asarray(frame.convert('RGB')) for frame in frames]
    imageio.mimsave(
        str(path),
        arrays,
        fps=fps,
        codec='libx264',
        quality=8,
        macro_block_size=1,
    )


def save_side_by_side(left: List[Image.Image], right: List[Image.Image],
                      path: Path, fps: int) -> None:
    count = min(len(left), len(right))
    frames = []
    for i in range(count):
        a = left[i].convert('RGB')
        b = right[i].convert('RGB')
        canvas = Image.new('RGB', (a.width + b.width, max(a.height, b.height)))
        canvas.paste(a, (0, 0))
        canvas.paste(b, (a.width, 0))
        frames.append(canvas)
    save_video(frames, path, fps)


def build_sample(cfg: Config, stats: Dict, sample_index: int):
    dataset_cfg = cfg.train_dataloader.dataset.copy()
    dataset_cfg.shuffle = False
    dataset_cfg.dataset_statistics = stats
    dataset = build_dataset_from_cfg(dataset_cfg)
    return dataset._get_item_from_global_idx(sample_index)


def build_dataset(cfg: Config, stats: Dict):
    dataset_cfg = cfg.train_dataloader.dataset.copy()
    dataset_cfg.shuffle = False
    dataset_cfg.dataset_statistics = stats
    return build_dataset_from_cfg(dataset_cfg)


def episode_bounds(dataset, sample_index: int):
    inner = dataset.dataset
    raw = inner.dataset
    raw_index = int(inner._resolve_index(sample_index))
    episode_index = raw[raw_index]['episode_index']

    start = raw_index
    while start > 0 and raw[start - 1]['episode_index'] == episode_index:
        start -= 1

    end = raw_index
    raw_len = len(raw)
    while end + 1 < raw_len and raw[end + 1]['episode_index'] == episode_index:
        end += 1

    return int(episode_index), start, end


def build_model(cfg: Config,
                ckpt_path: str,
                device: str,
                use_cached_context: bool = False):
    model_cfg = cfg.inference_model.copy()
    model_cfg.torch_dtype = 'bf16'
    if use_cached_context:
        model_cfg.vlm_backbone = None
    model = build_vla_from_cfg(model_cfg).eval()
    state_dict = load_file(ckpt_path, device='cpu')
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    gc.collect()
    if torch.cuda.is_available() and device.startswith('cuda'):
        torch.cuda.empty_cache()
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    model.torch_dtype = torch.bfloat16
    if hasattr(model, 'vla_head'):
        model.vla_head.torch_dtype = torch.bfloat16
    return model


def sample_to_inputs(sample: Dict):
    inputs = dict(
        input_image=batch_tensor(sample['images'][:, 0], dtype=torch.float32),
        gt_frames=normalized_video_to_frames(sample['images']),
        gt_action=torch.as_tensor(sample['actions'], dtype=torch.float32),
        proprio=torch.as_tensor(sample['states'], dtype=torch.float32),
        lang_tokens=None,
        lang_masks=None,
        context=None,
        context_mask=None,
    )
    if sample.get('lang_tokens') is not None:
        inputs['lang_tokens'] = batch_tensor(
            sample['lang_tokens'], dtype=torch.long)
    if sample.get('lang_masks') is not None:
        inputs['lang_masks'] = batch_tensor(
            sample['lang_masks'], dtype=torch.bool)
    if sample.get('context') is not None:
        context = torch.as_tensor(sample['context'], dtype=torch.float32)
        if context.ndim == 2:
            context = context.unsqueeze(0)
        inputs['context'] = context
    if sample.get('context_mask') is not None:
        context_mask = torch.as_tensor(
            sample['context_mask'], dtype=torch.bool)
        if context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)
        inputs['context_mask'] = context_mask
    return inputs


def condition_kwargs(inputs: Dict):
    if inputs.get('context') is not None:
        return dict(
            context=inputs['context'],
            context_mask=inputs['context_mask'],
        )
    if inputs.get('lang_tokens') is None or inputs.get('lang_masks') is None:
        raise ValueError('Sample must provide either `context/context_mask` '
                         'or `lang_tokens/lang_masks`.')
    return dict(
        lang_tokens=inputs['lang_tokens'],
        lang_masks=inputs['lang_masks'],
    )


@torch.no_grad()
def run_gt_action_forward_episode(args, cfg: Config, model, dataset,
                                  output_dir: Path, sample_index: int,
                                  frame_sample_stride: int):
    episode_index, episode_start, episode_end = episode_bounds(
        dataset, sample_index)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'[open-loop] gt_action_forward_output_dir={output_dir}')

    policy_step_rows = max(1, int(cfg._action_horizon))
    policy_last_frame_index = int(cfg._frame_window_size) - 1

    start_sample = dataset._get_item_from_global_idx(episode_start)
    start_inputs = sample_to_inputs(start_sample)
    gt_frames = [start_inputs['gt_frames'][0]]
    forward_frames = [start_inputs['gt_frames'][0]]

    chunks = 0
    t0 = time.time()
    cursor = episode_start
    while cursor < episode_end:
        valid_policy_idx = min(policy_last_frame_index,
                               (episode_end - cursor) // frame_sample_stride)
        if valid_policy_idx <= 0:
            break

        sample = dataset._get_item_from_global_idx(cursor)
        inputs = sample_to_inputs(sample)
        gt_frames.extend(inputs['gt_frames'][1:valid_policy_idx + 1])

        chunk = model.infer(
            action=inputs['gt_action'],
            input_image=inputs['input_image'],
            num_frames=int(cfg._frame_window_size),
            proprio=inputs['proprio'],
            num_inference_steps=args.num_inference_steps,
            seed=args.seed + 20000 + chunks,
            rand_device='cpu',
            **condition_kwargs(inputs),
        )
        forward_frames.extend(chunk['video'][1:valid_policy_idx + 1])
        chunks += 1
        cursor += policy_step_rows
        print(f'[open-loop] gt-action-forward episode chunk {chunks} '
              f'raw_index={cursor}')

    print(f'[open-loop] gt-action-forward whole episode done in '
          f'{time.time() - t0:.1f}s')

    save_video(gt_frames, output_dir / 'episode_gt_video.mp4', args.fps)
    save_video(forward_frames,
               output_dir / 'gt_action_then_forward_episode.mp4', args.fps)
    save_side_by_side(
        gt_frames, forward_frames,
        output_dir / 'compare_gt_vs_gt_action_forward_episode.mp4', args.fps)

    metrics = {
        'sample_index':
        sample_index,
        'episode_index':
        episode_index,
        'episode_start_index':
        episode_start,
        'episode_end_index':
        episode_end,
        'episode_rows':
        episode_end - episode_start + 1,
        'num_inference_steps':
        args.num_inference_steps,
        'frame_sample_stride':
        frame_sample_stride,
        'policy_step_rows':
        policy_step_rows,
        'policy_frame_window_size':
        int(cfg._frame_window_size),
        'gt_action_forward_chunks':
        chunks,
        'saved_frames':
        len(forward_frames),
        'fps':
        args.fps,
        'outputs': [
            'episode_gt_video.mp4',
            'gt_action_then_forward_episode.mp4',
            'compare_gt_vs_gt_action_forward_episode.mp4',
        ],
    }
    with (output_dir / 'gt_action_forward_metrics.json').open(
            'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


@torch.no_grad()
def run_whole_episode(args, cfg: Config, model, dataset, flux_stats: Dict,
                      output_dir: Path, sample_index: int,
                      next_action_index: int, frame_sample_stride: int):
    episode_index, episode_start, episode_end = episode_bounds(
        dataset, sample_index)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'[open-loop] whole_episode_output_dir={output_dir}')

    next_action_offset = int(args.next_frame_index) * int(frame_sample_stride)
    ar_step_rows = max(1, next_action_offset)
    policy_step_rows = max(1, int(cfg._action_horizon))
    next_frame_index = int(args.next_frame_index)
    policy_last_frame_index = int(cfg._frame_window_size) - 1

    start_sample = dataset._get_item_from_global_idx(episode_start)
    start_inputs = sample_to_inputs(start_sample)
    policy_gt_frames = [start_inputs['gt_frames'][0]]
    ar_gt_frames = [start_inputs['gt_frames'][0]]
    teacher_frames = [start_inputs['gt_frames'][0]]
    ar_frames = [start_inputs['gt_frames'][0]]

    policy_chunks = 0
    policy_action_mse = []
    t0 = time.time()
    cursor = episode_start
    while (not args.ar_only) and cursor < episode_end:
        valid_policy_idx = min(policy_last_frame_index,
                               (episode_end - cursor) // frame_sample_stride)
        if valid_policy_idx <= 0:
            break

        sample = dataset._get_item_from_global_idx(cursor)
        inputs = sample_to_inputs(sample)
        policy_gt_frames.extend(inputs['gt_frames'][1:valid_policy_idx + 1])

        pred_action = model.predict_action(
            input_image=inputs['input_image'],
            proprio=inputs['proprio'],
            num_inference_steps=args.num_inference_steps,
            seed=args.seed + policy_chunks,
            rand_device='cpu',
            **condition_kwargs(inputs),
        )[0].detach().cpu().float()
        chunk = model.infer(
            action=pred_action,
            input_image=inputs['input_image'],
            num_frames=int(cfg._frame_window_size),
            proprio=inputs['proprio'],
            num_inference_steps=args.num_inference_steps,
            seed=args.seed + 10000 + policy_chunks,
            rand_device='cpu',
            **condition_kwargs(inputs),
        )
        teacher_frames.extend(chunk['video'][1:valid_policy_idx + 1])
        pred_action_np = pred_action.numpy()
        gt_action_np = inputs['gt_action'].numpy()
        eval_dim = active_action_dim(pred_action_np, flux_stats)
        action_diff = (
            pred_action_np[..., :eval_dim] - gt_action_np[..., :eval_dim])
        policy_action_mse.append(float(np.mean(action_diff * action_diff)))
        policy_chunks += 1
        cursor += policy_step_rows
        print(f'[open-loop] policy-forward episode chunk {policy_chunks} '
              f'raw_index={cursor}')

    if args.ar_only:
        print('[open-loop] skip policy-forward whole episode')
    else:
        print(f'[open-loop] policy-forward whole episode done in '
              f'{time.time() - t0:.1f}s')

    ar_chunks = 0
    t0 = time.time()
    cursor = episode_start
    current_image = start_inputs['input_image']
    current_proprio = start_inputs['proprio']
    while cursor < episode_end:
        valid_next_idx = min(next_frame_index,
                             (episode_end - cursor) // frame_sample_stride)
        if valid_next_idx <= 0:
            break

        sample = dataset._get_item_from_global_idx(cursor)
        inputs = sample_to_inputs(sample)
        ar_gt_frames.extend(inputs['gt_frames'][1:valid_next_idx + 1])

        chunk_action = model.predict_action(
            input_image=current_image,
            proprio=current_proprio,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed + ar_chunks,
            rand_device='cpu',
            **condition_kwargs(start_inputs),
        )[0].detach().cpu().float()
        chunk = model.infer(
            action=chunk_action,
            input_image=current_image,
            num_frames=int(cfg._frame_window_size),
            proprio=current_proprio,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed + 10000 + ar_chunks,
            rand_device='cpu',
            **condition_kwargs(start_inputs),
        )

        frames = chunk['video']
        ar_frames.extend(frames[1:valid_next_idx + 1])
        next_frame = np.asarray(frames[valid_next_idx].convert('RGB')).astype(
            np.float32)
        next_frame = torch.from_numpy(next_frame).permute(2, 0, 1)
        next_frame = next_frame / 127.5 - 1.0
        current_image = next_frame.unsqueeze(0)

        state_action_index = min(next_action_index,
                                 valid_next_idx * frame_sample_stride)
        next_state = denormalize_action(
            chunk_action[state_action_index].numpy(), flux_stats)
        current_proprio = torch.from_numpy(
            normalize_state(next_state, flux_stats)).float()

        ar_chunks += 1
        cursor += ar_step_rows
        print(f'[open-loop] autoregressive episode chunk {ar_chunks} '
              f'raw_index={cursor}')

    print(f'[open-loop] autoregressive whole episode done in '
          f'{time.time() - t0:.1f}s')

    save_video(ar_gt_frames, output_dir / 'episode_gt_video.mp4', args.fps)
    if not args.ar_only:
        save_video(teacher_frames,
                   output_dir / 'policy_action_then_forward_episode.mp4',
                   args.fps)
    save_video(ar_frames,
               output_dir / 'autoregressive_policy_action_forward_episode.mp4',
               args.fps)
    if not args.ar_only:
        save_side_by_side(
            policy_gt_frames, teacher_frames,
            output_dir / 'compare_gt_vs_policy_action_forward_episode.mp4',
            args.fps)
    save_side_by_side(
        ar_gt_frames,
        ar_frames,
        output_dir /
        'compare_gt_vs_autoregressive_policy_action_forward_episode.mp4',  # noqa: E501
        args.fps)

    metrics = {
        'sample_index':
        sample_index,
        'episode_index':
        episode_index,
        'episode_start_index':
        episode_start,
        'episode_end_index':
        episode_end,
        'episode_rows':
        episode_end - episode_start + 1,
        'num_inference_steps':
        args.num_inference_steps,
        'next_frame_index':
        next_frame_index,
        'next_action_offset':
        next_action_offset,
        'next_action_index':
        next_action_index,
        'frame_sample_stride':
        frame_sample_stride,
        'policy_step_rows':
        policy_step_rows,
        'policy_frame_window_size':
        int(cfg._frame_window_size),
        'ar_step_rows':
        ar_step_rows,
        'policy_forward_chunks':
        policy_chunks,
        'autoregressive_chunks':
        ar_chunks,
        'saved_frames':
        len(ar_frames),
        'fps':
        args.fps,
        'policy_forward_action_mse_mean':
        float(np.mean(policy_action_mse)) if policy_action_mse else None,
        'autoregressive_order':
        'predict_action -> forward_dynamics(image,state,action) -> '
        'action_as_next_proprio',
        'forward_action_conditioned':
        True,
        'outputs': [
            'episode_gt_video.mp4',
            'autoregressive_policy_action_forward_episode.mp4',
            'compare_gt_vs_autoregressive_policy_action_forward_episode.mp4',
        ],
    }
    if not args.ar_only:
        metrics['outputs'].insert(1, 'policy_action_then_forward_episode.mp4')
        metrics['outputs'].insert(
            3, 'compare_gt_vs_policy_action_forward_episode.mp4')
    with (output_dir / 'metrics.json').open('w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


@torch.no_grad()
def main():
    args = parse_args()
    configure_inference_attention_defaults()

    ckpt_path = Path(args.ckpt_path)
    work_dir = ckpt_path.resolve().parent.parent
    stats_path = work_dir / 'dataset_statistics.json'
    cfg = Config.fromfile(args.config)
    statistic_name = cfg.train_dataloader.dataset.statistic_name
    stats = load_dataset_stats(stats_path, statistic_name)
    flux_stats = stats[statistic_name]

    output_dir = (
        Path(args.output_dir) if args.output_dir is not None else
        _default_output_dir(args.ckpt_path, args.sample_index))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'[open-loop] sample_index={args.sample_index}')
    print(f'[open-loop] checkpoint={ckpt_path}')
    print(f'[open-loop] output_dir={output_dir}')
    print(f'[open-loop] stats={stats_path}')

    sample = build_sample(cfg, stats, args.sample_index)
    inputs = sample_to_inputs(sample)
    gt_frames = inputs['gt_frames']
    save_video(gt_frames, output_dir / 'episode_gt_video.mp4', args.fps)

    input_image = inputs['input_image']
    gt_action = inputs['gt_action']
    proprio = inputs['proprio']
    condition = condition_kwargs(inputs)

    model = build_model(
        cfg,
        str(ckpt_path),
        args.device,
        use_cached_context=inputs.get('context') is not None,
    )

    frame_sample_stride = int(cfg._frame_sample_stride)
    next_action_index = max(
        0,
        min(
            int(args.next_frame_index) * frame_sample_stride,
            int(cfg._action_horizon) - 1),
    )
    if args.whole_episode:
        dataset = build_dataset(cfg, stats)
        episode_index, _, _ = episode_bounds(dataset, args.sample_index)
        if args.output_dir is None:
            output_dir = work_dir / (
                f'open_loop_aloha_episode_{episode_index:06d}_'
                'policy_action_forward_ar')
        if args.gt_action_forward_only:
            run_gt_action_forward_episode(
                args=args,
                cfg=cfg,
                model=model,
                dataset=dataset,
                output_dir=output_dir,
                sample_index=args.sample_index,
                frame_sample_stride=frame_sample_stride,
            )
            return
        run_whole_episode(
            args=args,
            cfg=cfg,
            model=model,
            dataset=dataset,
            flux_stats=flux_stats,
            output_dir=output_dir,
            sample_index=args.sample_index,
            next_action_index=next_action_index,
            frame_sample_stride=frame_sample_stride,
        )
        return

    t0 = time.time()
    pred_action = model.predict_action(
        input_image=input_image,
        proprio=proprio,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        rand_device='cpu',
        **condition,
    )[0].detach().cpu().float()
    first_forward = model.infer(
        action=pred_action,
        input_image=input_image,
        num_frames=int(cfg._frame_window_size),
        proprio=proprio,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed + 10000,
        rand_device='cpu',
        **condition,
    )
    first_forward_frames = first_forward['video']
    save_video(first_forward_frames,
               output_dir / 'policy_action_then_forward.mp4', args.fps)
    save_side_by_side(gt_frames, first_forward_frames,
                      output_dir / 'compare_gt_vs_policy_action_forward.mp4',
                      args.fps)
    print(f'[open-loop] policy_action_then_forward done in '
          f'{time.time() - t0:.1f}s')

    action_np = pred_action.numpy()
    gt_action_np = gt_action.numpy()
    pred_denorm = denormalize_action(action_np, flux_stats)
    gt_denorm = denormalize_action(gt_action_np, flux_stats)
    action_eval_dim = active_action_dim(action_np, flux_stats)
    action_diff = (
        action_np[..., :action_eval_dim] - gt_action_np[..., :action_eval_dim])
    denorm_diff = (
        pred_denorm[..., :action_eval_dim] - gt_denorm[..., :action_eval_dim])

    rollout_frames = []
    current_image = input_image
    current_proprio = proprio
    for chunk_idx in range(args.rollout_chunks):
        chunk_seed = args.seed + chunk_idx
        chunk_action = model.predict_action(
            input_image=current_image,
            proprio=current_proprio,
            num_inference_steps=args.num_inference_steps,
            seed=chunk_seed,
            rand_device='cpu',
            **condition,
        )[0].detach().cpu().float()
        next_state = denormalize_action(
            chunk_action[next_action_index].numpy(), flux_stats)
        next_proprio = torch.from_numpy(
            normalize_state(next_state, flux_stats)).float()
        chunk = model.infer(
            action=chunk_action,
            input_image=current_image,
            num_frames=int(cfg._frame_window_size),
            proprio=current_proprio,
            num_inference_steps=args.num_inference_steps,
            seed=chunk_seed + 10000,
            rand_device='cpu',
            **condition,
        )
        frames = chunk['video']
        next_idx = max(0, min(args.next_frame_index, len(frames) - 1))
        if chunk_idx == 0:
            rollout_frames.append(frames[0])
        rollout_frames.extend(frames[1:next_idx + 1])
        next_frame = np.asarray(frames[next_idx].convert('RGB')).astype(
            np.float32)
        next_frame = torch.from_numpy(next_frame).permute(2, 0, 1)
        next_frame = next_frame / 127.5 - 1.0
        current_image = next_frame.unsqueeze(0)
        current_proprio = next_proprio
        print(f'[open-loop] autoregressive chunk '
              f'{chunk_idx + 1}/{args.rollout_chunks} done')

    save_video(rollout_frames,
               output_dir / 'autoregressive_policy_action_forward.mp4',
               args.fps)

    metrics = {
        'sample_index':
        args.sample_index,
        'task_description':
        sample.get('task_description', ''),
        'num_inference_steps':
        args.num_inference_steps,
        'rollout_chunks':
        args.rollout_chunks,
        'next_frame_index':
        args.next_frame_index,
        'next_action_index':
        next_action_index,
        'action_eval_dim':
        action_eval_dim,
        'proprio_stat_dim':
        len(flux_stats['proprio']['mean']),
        'action_stat_dim':
        len(flux_stats['action']['mean']),
        'frame_sample_stride':
        frame_sample_stride,
        'autoregressive_order':
        'predict_action -> forward_dynamics(image,state,action) -> '
        'action_as_next_proprio',
        'forward_action_conditioned':
        True,
        'normalized_action_mse':
        float(np.mean(action_diff * action_diff)),
        'denormalized_action_mse':
        float(np.mean(denorm_diff * denorm_diff)),
        'denormalized_action_mae':
        float(np.mean(np.abs(denorm_diff))),
        'pred_action_mean_abs':
        float(np.mean(np.abs(action_np[..., :action_eval_dim]))),
        'gt_action_mean_abs':
        float(np.mean(np.abs(gt_action_np[..., :action_eval_dim]))),
        'outputs': [
            'episode_gt_video.mp4',
            'policy_action_then_forward.mp4',
            'autoregressive_policy_action_forward.mp4',
            'compare_gt_vs_policy_action_forward.mp4',
        ],
    }
    with (output_dir / 'metrics.json').open('w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
