import argparse
import contextlib
import csv
import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import imageio
import numpy as np
import torch
from einops import rearrange
from mmengine import Config, DictAction
from safetensors import safe_open
from safetensors.torch import load_file
from torch.utils.data import DataLoader

import fluxvla  # noqa: F401
from fluxvla.engines import (build_collator_from_cfg, build_dataset_from_cfg,
                             build_vla_from_cfg)

DEFAULT_CONFIG = 'configs/dreamzero/dreamzero_hud04_full_finetune.py'
DEFAULT_WORK_DIR = '/mnt/data/cpfs/users/jikun/wk_dir/dreamzero_hud04'


def log(message: str) -> None:
    print(f'[check_dreamzero_predictions] {message}', flush=True)


@contextlib.contextmanager
def skip_random_init(enabled: bool = True):
    if not enabled:
        yield
        return

    init_fns = [
        'constant_', 'kaiming_normal_', 'kaiming_uniform_', 'normal_', 'ones_',
        'trunc_normal_', 'uniform_', 'xavier_normal_', 'xavier_uniform_',
        'zeros_'
    ]
    originals = {}

    def no_init(tensor, *args, **kwargs):
        return tensor

    for name in init_fns:
        if hasattr(torch.nn.init, name):
            originals[name] = getattr(torch.nn.init, name)
            setattr(torch.nn.init, name, no_init)
    try:
        yield
    finally:
        for name, fn in originals.items():
            setattr(torch.nn.init, name, fn)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Check DreamZero checkpoint predictions on config data.')
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--work-dir', default=DEFAULT_WORK_DIR)
    parser.add_argument('--ckpt-path', default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--num-samples-per-embodiment', type=int, default=1)
    parser.add_argument(
        '--segments-per-sample',
        type=int,
        default=1,
        help='Number of consecutive dataset items to roll out per sample.')
    parser.add_argument(
        '--segment-stride',
        type=int,
        default=None,
        help='Dataset index stride between rollout segments. Defaults to the '
        'DreamZero RGB-frame prediction horizon, num_frame_per_block * 4.')
    parser.add_argument('--max-batches', type=int, default=200)
    parser.add_argument('--num-inference-steps', type=int, default=None)
    parser.add_argument('--fps', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument(
        '--model-dtype',
        default='bf16',
        choices=['bf16', 'fp32'],
        help='Parameter dtype used when constructing the model.')
    parser.add_argument(
        '--no-skip-random-init',
        action='store_true',
        help='Do not monkeypatch torch.nn.init during model construction.')
    parser.add_argument(
        '--disable-video-pred',
        action='store_true',
        help='Only save input video and action predictions.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config options, e.g. model.use_cache=False')
    return parser.parse_args()


def resolve_ckpt_path(work_dir: str, ckpt_path: Optional[str]) -> str:
    if ckpt_path is not None:
        return ckpt_path
    candidate = Path(
        work_dir) / 'checkpoints' / 'latest-checkpoint.safetensors'
    if candidate.exists():
        return str(candidate)
    candidate = Path(work_dir) / 'checkpoints' / 'latest-checkpoint.pt'
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(f'No latest checkpoint found under {work_dir}')


def load_checkpoint_state_dict(ckpt_path: str) -> Dict[str, torch.Tensor]:
    ckpt_path_obj = Path(ckpt_path)
    if ckpt_path_obj.suffix == '.pt':
        sf_candidate = ckpt_path_obj.with_suffix('.safetensors')
        if sf_candidate.exists():
            ckpt_path_obj = sf_candidate

    if ckpt_path_obj.suffix == '.safetensors':
        return load_file(str(ckpt_path_obj), device='cpu')

    try:
        checkpoint = torch.load(
            str(ckpt_path_obj), map_location='cpu', mmap=True)
    except TypeError:
        checkpoint = torch.load(str(ckpt_path_obj), map_location='cpu')
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    return state_dict


def load_safetensors_streaming(model: torch.nn.Module,
                               ckpt_path: str) -> tuple[List[str], List[str]]:
    model_state = model.state_dict()
    loaded_keys = set()
    unexpected = []
    with safe_open(ckpt_path, framework='pt', device='cpu') as file:
        for key in file.keys():
            if key not in model_state:
                unexpected.append(key)
                continue
            tensor = file.get_tensor(key)
            target = model_state[key]
            if tensor.shape != target.shape:
                raise RuntimeError(
                    f'Shape mismatch for {key}: ckpt {tuple(tensor.shape)} '
                    f'vs model {tuple(target.shape)}')
            with torch.no_grad():
                target.copy_(tensor.to(dtype=target.dtype))
            loaded_keys.add(key)
    missing = [key for key in model_state.keys() if key not in loaded_keys]
    if missing or unexpected:
        raise RuntimeError(f'Checkpoint load mismatch. missing={missing[:20]} '
                           f'unexpected={unexpected[:20]}')
    return missing, unexpected


def load_checkpoint_into_model(model: torch.nn.Module,
                               ckpt_path: str) -> tuple[List[str], List[str]]:
    ckpt_path_obj = Path(ckpt_path)
    if ckpt_path_obj.suffix == '.pt':
        sf_candidate = ckpt_path_obj.with_suffix('.safetensors')
        if sf_candidate.exists():
            ckpt_path_obj = sf_candidate
    if ckpt_path_obj.suffix == '.safetensors':
        return load_safetensors_streaming(model, str(ckpt_path_obj))

    state_dict = load_checkpoint_state_dict(str(ckpt_path_obj))
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    del state_dict
    gc.collect()
    return missing, unexpected


def tensor_to_device(batch: Dict[str, Any],
                     device: torch.device) -> Dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def save_video_from_normalized_tensor(video: torch.Tensor, save_path: Path,
                                      fps: int) -> str:
    frames = normalized_video_to_frames(video)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(save_path, list(frames), fps=fps, codec='libx264')
    return str(save_path)


def normalized_video_to_frames(video: torch.Tensor) -> np.ndarray:
    video = video.detach().float().cpu()
    if video.ndim == 5:
        video = video[0]
    frames = rearrange(video, 'c t h w -> t h w c')
    frames = ((frames + 1.0) * 127.5).clamp(0, 255).numpy().astype('uint8')
    return frames


def save_frames_as_video(frames: np.ndarray, save_path: Path, fps: int) -> str:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(save_path, list(frames), fps=fps, codec='libx264')
    return str(save_path)


def read_video_frames(video_path: str) -> List[np.ndarray]:
    return [frame for frame in imageio.mimread(video_path)]


def resize_to_height(frame: np.ndarray, target_height: int) -> np.ndarray:
    if frame.shape[0] == target_height:
        return frame
    scale = target_height / frame.shape[0]
    target_width = max(1, int(round(frame.shape[1] * scale)))
    return cv2.resize(
        frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def add_label(frame: np.ndarray, label: str) -> np.ndarray:
    header_h = 32
    labeled = np.zeros((frame.shape[0] + header_h, frame.shape[1], 3),
                       dtype=np.uint8)
    labeled[header_h:] = frame
    cv2.putText(labeled, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return labeled


def save_side_by_side_video(gt_video_path: str, pred_video_path: str,
                            save_path: Path, fps: int) -> str:
    gt_frames = read_video_frames(gt_video_path)
    pred_frames = read_video_frames(pred_video_path)
    if len(gt_frames) == 0 or len(pred_frames) == 0:
        raise ValueError('GT and pred videos must both contain frames.')

    num_frames = max(len(gt_frames), len(pred_frames))
    combined_frames = []
    for frame_index in range(num_frames):
        gt_index = min(
            len(gt_frames) - 1,
            int(
                round(frame_index * (len(gt_frames) - 1) /
                      max(num_frames - 1, 1))))
        pred_index = min(
            len(pred_frames) - 1,
            int(
                round(frame_index * (len(pred_frames) - 1) /
                      max(num_frames - 1, 1))))
        gt_frame = gt_frames[gt_index]
        pred_frame = pred_frames[pred_index]
        target_height = max(gt_frame.shape[0], pred_frame.shape[0])
        gt_frame = resize_to_height(gt_frame, target_height)
        pred_frame = resize_to_height(pred_frame, target_height)
        gt_frame = add_label(gt_frame, 'GT')
        pred_frame = add_label(pred_frame, 'PRED')
        combined_frames.append(np.concatenate([gt_frame, pred_frame], axis=1))

    return save_frames_as_video(np.stack(combined_frames), save_path, fps)


def append_sliding_window_frames(frame_windows: List[np.ndarray],
                                 window_frames: np.ndarray,
                                 new_frame_count: int) -> None:
    """Append a sliding-window video without duplicating overlap.

    Consecutive dataset items are offset by one timestep but each item contains
    the full frame window, e.g. [0..8], [1..9], [2..10]. For a long GT clip we
    keep the first full window and then append only the newly revealed tail
    frame from each subsequent window.
    """
    if len(frame_windows) == 0:
        frame_windows.append(window_frames)
        return
    new_frame_count = max(1, min(new_frame_count, window_frames.shape[0]))
    frame_windows.append(window_frames[-new_frame_count:])


def stats_to_numpy(stats: Dict[str, Any],
                   key: str) -> Optional[Dict[str, np.ndarray]]:
    if key not in stats:
        return None
    return {
        stat_key: np.asarray(stat_value, dtype=np.float32)
        for stat_key, stat_value in stats[key].items()
        if stat_key in ('mean', 'std', 'min', 'max', 'q01', 'q99')
    }


def denormalize_mean_std(values: np.ndarray,
                         stats: Optional[Dict[str, np.ndarray]]) -> np.ndarray:
    if stats is None or 'mean' not in stats or 'std' not in stats:
        return values.copy()
    dim = min(values.shape[-1], stats['mean'].shape[-1])
    restored = values.copy()
    restored[..., :dim] = restored[..., :dim] * (stats['std'][:dim] + 1e-6)
    restored[..., :dim] = restored[..., :dim] + stats['mean'][:dim]
    return restored


def write_action_csv(csv_path: Path, sample_name: str, pred_norm: np.ndarray,
                     target_norm: np.ndarray, pred_raw: np.ndarray,
                     target_raw: np.ndarray, masks: np.ndarray) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'sample', 'step', 'dim', 'mask', 'pred_norm', 'target_norm',
            'pred_denorm', 'target_denorm', 'abs_err_norm', 'abs_err_denorm'
        ])
        horizon, action_dim = pred_norm.shape
        for step in range(horizon):
            mask = float(masks[step]) if masks.ndim == 1 else float(
                masks[step].max())
            for dim in range(action_dim):
                writer.writerow([
                    sample_name,
                    step,
                    dim,
                    mask,
                    float(pred_norm[step, dim]),
                    float(target_norm[step, dim]),
                    float(pred_raw[step, dim]),
                    float(target_raw[step, dim]),
                    float(abs(pred_norm[step, dim] - target_norm[step, dim])),
                    float(abs(pred_raw[step, dim] - target_raw[step, dim])),
                ])


def summarize_actions(pred_norm: np.ndarray, target_norm: np.ndarray,
                      pred_raw: np.ndarray, target_raw: np.ndarray,
                      masks: np.ndarray) -> Dict[str, float]:
    valid = masks.astype(bool)
    if valid.ndim != 1:
        valid = valid.any(axis=-1)
    if not valid.any():
        valid = np.ones(pred_norm.shape[0], dtype=bool)
    return {
        'mae_norm':
        float(np.abs(pred_norm[valid] - target_norm[valid]).mean()),
        'max_abs_err_norm':
        float(np.abs(pred_norm[valid] - target_norm[valid]).max()),
        'mae_denorm':
        float(np.abs(pred_raw[valid] - target_raw[valid]).mean()),
        'max_abs_err_denorm':
        float(np.abs(pred_raw[valid] - target_raw[valid]).max()),
    }


def iter_sample_sequences(dataset, collator, max_batches: int,
                          num_samples_per_group: int, segments_per_sample: int,
                          segment_stride: int):
    if getattr(dataset, 'is_grouped', False):
        group_start = 0
        batch_index = 0
        grouped_lens = dataset.grouped_cumulative_lens
        for group_name, cumulative_lens in grouped_lens.items():
            group_total = cumulative_lens[-1]
            for sample_offset in range(num_samples_per_group):
                start_offset = (
                    sample_offset * segments_per_sample * segment_stride)
                if start_offset >= group_total:
                    break
                batches = []
                for segment_index in range(segments_per_sample):
                    local_offset = (
                        start_offset + segment_index * segment_stride)
                    if local_offset >= group_total:
                        break
                    sample = dataset._get_item_from_global_idx(group_start +
                                                               local_offset)
                    batches.append(collator([sample]))
                if batches:
                    yield batch_index, group_name, batches
                    batch_index += 1
            group_start += group_total
        return

    dataloader = DataLoader(
        dataset, batch_size=1, collate_fn=collator, num_workers=0)
    iterator = iter(dataloader)
    batch_index = 0
    while batch_index < max_batches:
        batches = []
        for segment_index in range(segments_per_sample):
            try:
                batch = next(iterator)
                for _ in range(segment_stride - 1):
                    next(iterator)
                batches.append(batch)
            except StopIteration:
                break
        if not batches:
            break
        yield batch_index, None, batches
        batch_index += 1


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.segment_stride is None:
        args.segment_stride = int(cfg.model.vla_head.num_frame_per_block) * 4

    ckpt_path = resolve_ckpt_path(args.work_dir, args.ckpt_path)
    output_dir = Path(
        args.output_dir or Path(args.work_dir) / 'prediction_check')
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f'Using checkpoint: {ckpt_path}')
    log(f'Writing outputs to: {output_dir}')

    if args.device == 'cuda' and not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    cfg.train_dataloader.dataset.shuffle = False
    cfg.train_dataloader.per_device_batch_size = 1
    cfg.train_dataloader.per_device_num_workers = 0

    log('Building dataset...')
    dataset = build_dataset_from_cfg(cfg.train_dataloader.dataset)
    collator = build_collator_from_cfg(cfg.runner.collator)

    model_dtype = (
        torch.bfloat16 if args.model_dtype == 'bf16' else torch.float32)
    previous_default_dtype = torch.get_default_dtype()
    log(f'Building model with default dtype {model_dtype}...')
    torch.set_default_dtype(model_dtype)
    try:
        with skip_random_init(not args.no_skip_random_init):
            model = build_vla_from_cfg(cfg.model).eval()
    finally:
        torch.set_default_dtype(previous_default_dtype)
    log('Loading checkpoint weights...')
    missing, unexpected = load_checkpoint_into_model(model, ckpt_path)
    log(f'Moving model to {device}...')
    model.to(device)
    if args.num_inference_steps is not None:
        model.vla_head.num_inference_steps = args.num_inference_steps
    if hasattr(model, 'enable_video_prediction'):
        model.enable_video_prediction(not args.disable_video_pred)

    action_dim = int(getattr(model.vla_head, 'action_dim', 52))
    seen_by_embodiment: Dict[int, int] = {}
    summaries: List[Dict[str, Any]] = []

    autocast_enabled = device.type == 'cuda'
    autocast_dtype = torch.bfloat16

    with torch.inference_mode():
        for batch_index, group_name, batches in iter_sample_sequences(
                dataset, collator, args.max_batches,
                args.num_samples_per_embodiment, args.segments_per_sample,
                args.segment_stride):

            first_batch = batches[0]
            embodiment_id = int(
                first_batch.get('embodiment_ids', torch.zeros(1))[0].item())
            if seen_by_embodiment.get(embodiment_id,
                                      0) >= args.num_samples_per_embodiment:
                continue

            sample_id = len(summaries)
            sample_name = f'sample_{sample_id:03d}_emb{embodiment_id}'
            log(f'Running {sample_name} group={group_name} '
                f'batch={batch_index} segments={len(batches)}')
            sample_dir = output_dir / sample_name
            sample_dir.mkdir(parents=True, exist_ok=True)

            if hasattr(model, 'reset_video_prediction'):
                model.reset_video_prediction()

            gt_frames = []
            pred_actions_by_segment = []
            target_actions_by_segment = []
            masks_by_segment = []

            for segment_index, batch in enumerate(batches):
                append_sliding_window_frames(
                    gt_frames, normalized_video_to_frames(batch['images']),
                    args.segment_stride)
                batch_on_device = tensor_to_device(batch, device)
                predict_kwargs = {
                    'images': batch_on_device['images'],
                    'lang_tokens': batch_on_device['lang_tokens'],
                    'lang_masks': batch_on_device['lang_masks'],
                    'states': batch_on_device['states'],
                    'embodiment_ids': batch_on_device.get('embodiment_ids'),
                    'reset_history': segment_index == 0,
                }

                with torch.autocast(
                        device_type=device.type,
                        dtype=autocast_dtype,
                        enabled=autocast_enabled):
                    pred_actions = model.predict_action(**predict_kwargs)

                pred_actions_by_segment.append(
                    pred_actions[0, :, :action_dim].float().cpu().numpy())
                target_actions_by_segment.append(
                    batch['actions'][0, :, :action_dim].float().cpu().numpy())
                masks_by_segment.append(
                    batch['action_masks'][0].float().cpu().numpy())

            gt_frames = np.concatenate(gt_frames, axis=0)
            gt_video_path = save_frames_as_video(gt_frames,
                                                 sample_dir / 'gt_video.mp4',
                                                 args.fps)
            input_video_path = gt_video_path

            pred_video_path = None
            side_by_side_video_path = None
            if not args.disable_video_pred and hasattr(
                    model, 'decode_and_save_video'):
                pred_video_path = model.decode_and_save_video(
                    str(sample_dir / 'pred_video.mp4'), fps=args.fps)
                if pred_video_path is not None:
                    side_by_side_video_path = save_side_by_side_video(
                        gt_video_path, pred_video_path,
                        sample_dir / 'gt_pred_side_by_side.mp4', args.fps)

            pred_norm = np.concatenate(pred_actions_by_segment, axis=0)
            target_norm = np.concatenate(target_actions_by_segment, axis=0)
            masks = np.concatenate(masks_by_segment, axis=0)
            stats = first_batch['stats'][0] if isinstance(
                first_batch['stats'], list) else first_batch['stats']
            action_stats = stats_to_numpy(stats, 'action')
            pred_raw = denormalize_mean_std(pred_norm, action_stats)
            target_raw = denormalize_mean_std(target_norm, action_stats)

            np.savez(
                sample_dir / 'actions.npz',
                pred_norm=pred_norm,
                target_norm=target_norm,
                pred_denorm=pred_raw,
                target_denorm=target_raw,
                action_masks=masks,
            )
            write_action_csv(sample_dir / 'actions.csv', sample_name,
                             pred_norm, target_norm, pred_raw, target_raw,
                             masks)

            action_summary = summarize_actions(pred_norm, target_norm,
                                               pred_raw, target_raw, masks)
            summary = {
                'sample': sample_name,
                'batch_index': batch_index,
                'group_name': group_name,
                'embodiment_id': embodiment_id,
                'segments': len(batches),
                'segment_stride': args.segment_stride,
                'task_description': first_batch.get('task_description',
                                                    [''])[0],
                'prompt': first_batch.get('prompt', [''])[0],
                'gt_video': gt_video_path,
                'input_video': input_video_path,
                'pred_video': pred_video_path,
                'side_by_side_video': side_by_side_video_path,
                'actions_csv': str(sample_dir / 'actions.csv'),
                'actions_npz': str(sample_dir / 'actions.npz'),
                'action_dim': action_dim,
                **action_summary,
            }
            summaries.append(summary)
            with (sample_dir / 'summary.json').open('w') as file:
                json.dump(summary, file, indent=2, ensure_ascii=False)
            log(f'Finished {sample_name}: mae_norm={summary["mae_norm"]:.6f}')

            seen_by_embodiment[embodiment_id] = seen_by_embodiment.get(
                embodiment_id, 0) + 1
            if len(seen_by_embodiment) >= 2 and all(
                    count >= args.num_samples_per_embodiment
                    for count in seen_by_embodiment.values()):
                break

    run_summary = {
        'config': args.config,
        'work_dir': args.work_dir,
        'ckpt_path': ckpt_path,
        'output_dir': str(output_dir),
        'device': str(device),
        'num_inference_steps': args.num_inference_steps,
        'segments_per_sample': args.segments_per_sample,
        'segment_stride': args.segment_stride,
        'missing_keys': missing,
        'unexpected_keys': unexpected,
        'samples': summaries,
    }
    with (output_dir / 'summary.json').open('w') as file:
        json.dump(run_summary, file, indent=2, ensure_ascii=False)

    print(json.dumps(run_summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
