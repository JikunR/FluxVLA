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

import glob
import inspect
import os
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from ...third_party_models.fastwam.modules.action_dit import ActionDiT
from ...third_party_models.fastwam.modules.wan_video_dit import WanVideoDiT
from ...third_party_models.fastwam.modules.wan_video_text_encoder import \
    WanTextEncoder
from ...third_party_models.fastwam.modules.wan_video_vae import WanVideoVAE38


def _validate_dit_config(dit_config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(dit_config, dict):
        raise ValueError(
            f'`dit_config` must be a dict, got {type(dit_config).__name__}')

    validated = dict(dit_config)
    signature = inspect.signature(WanVideoDiT.__init__)
    allowed_keys = set()
    required_keys = set()
    for name, param in signature.parameters.items():
        if name == 'self':
            continue
        allowed_keys.add(name)
        if param.default is inspect.Signature.empty:
            required_keys.add(name)

    unknown_keys = sorted(set(validated) - allowed_keys)
    if unknown_keys:
        raise ValueError(f'Unknown keys in `dit_config`: {unknown_keys}. '
                         f'Allowed keys: {sorted(allowed_keys)}')

    missing_keys = sorted(required_keys - set(validated))
    if missing_keys:
        raise ValueError(
            f'Missing required keys in `dit_config`: {missing_keys}. '
            'Please specify all required WanVideoDiT constructor args.')

    return validated


def _resolve_checkpoint_root(checkpoint_root: str) -> Path:
    root = Path(checkpoint_root).expanduser()
    if root.exists():
        return root

    base = Path(
        os.environ.get('DIFFSYNTH_MODEL_BASE_PATH',
                       os.environ.get('FLUXVLA_CHECKPOINT_ROOT', '')))
    if str(base):
        candidate = base / checkpoint_root
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f'Wan2.2 checkpoint root does not exist: {checkpoint_root}. Pass a '
        'local `checkpoint_root` path in the WAM config.')


def _glob_required(root: Path, pattern: str):
    matches = sorted(glob.glob(str(root / pattern)))
    if not matches:
        raise FileNotFoundError(
            f'Missing required Wan2.2 file pattern under {root}: {pattern}')
    return matches[0] if len(matches) == 1 else matches


def _glob_first_required(root: Path, patterns: tuple[str, ...]):
    for pattern in patterns:
        matches = sorted(glob.glob(str(root / pattern)))
        if matches:
            return matches[0] if len(matches) == 1 else matches
    raise FileNotFoundError(
        f'Missing required Wan2.2 file under {root}; tried patterns: '
        f'{list(patterns)}')


def _load_state_dict_from_safetensors(path: str,
                                      torch_dtype: torch.dtype | None = None):
    state_dict = {}
    with safe_open(path, framework='pt', device='cpu') as f:
        for key in f.keys():
            value = f.get_tensor(key)
            if torch_dtype is not None:
                value = value.to(torch_dtype)
            state_dict[key] = value
    return state_dict


def _load_state_dict_from_bin(path: str,
                              torch_dtype: torch.dtype | None = None):
    state_dict = torch.load(path, map_location='cpu', weights_only=True)
    if len(state_dict) == 1:
        state_dict = state_dict.get('state_dict', state_dict)
        state_dict = state_dict.get('module', state_dict)
        state_dict = state_dict.get('model_state', state_dict)
    if torch_dtype is not None:
        state_dict = {
            key:
            value.to(torch_dtype) if isinstance(value, torch.Tensor) else value
            for key, value in state_dict.items()
        }
    return state_dict


def _load_state_dict(path, torch_dtype: torch.dtype | None = None):
    if isinstance(path, list):
        merged = {}
        for part in path:
            merged.update(_load_state_dict(part, torch_dtype=torch_dtype))
        return merged
    if str(path).endswith('.safetensors'):
        return _load_state_dict_from_safetensors(path, torch_dtype)
    return _load_state_dict_from_bin(path, torch_dtype)


def _load_dit(path, dit_config: dict[str, Any], torch_dtype: torch.dtype,
              device: str):
    model = WanVideoDiT(**dit_config)
    state_dict = _load_state_dict(path, torch_dtype=torch_dtype)
    model.load_state_dict(state_dict, strict=False)
    return model.to(device=device, dtype=torch_dtype)


def _load_text_encoder(path, torch_dtype: torch.dtype, device: str):
    model = WanTextEncoder()
    state_dict = _load_state_dict(path, torch_dtype=torch_dtype)
    model.load_state_dict(state_dict, strict=False)
    return model.to(device=device, dtype=torch_dtype)


def _load_vae(path, torch_dtype: torch.dtype, device: str):
    model = WanVideoVAE38()
    state_dict = _load_state_dict(path, torch_dtype=torch_dtype)
    state_dict = {f'model.{key}': value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    return model.to(device=device, dtype=torch_dtype)


def build_wan_text_encoder(
    checkpoint_root: str | None = None,
    checkpoint_pattern: str | None = None,
    skip_load_from_pretrain: bool = False,
    device: str = 'cuda',
    torch_dtype: torch.dtype = torch.bfloat16,
):
    if skip_load_from_pretrain:
        return WanTextEncoder().to(device=device, dtype=torch_dtype)
    if checkpoint_root is None:
        raise ValueError(
            '`checkpoint_root` is required for WanTextEncoder unless '
            '`skip_load_from_pretrain=True`.')
    root = _resolve_checkpoint_root(checkpoint_root)
    pattern = checkpoint_pattern or 'models_t5_umt5-xxl-enc-bf16.*'
    path = _glob_required(root, pattern)
    return _load_text_encoder(path, torch_dtype=torch_dtype, device=device)


def build_wan_video_vae38(
    checkpoint_root: str | None = None,
    checkpoint_pattern: str | None = None,
    skip_load_from_pretrain: bool = False,
    device: str = 'cuda',
    torch_dtype: torch.dtype = torch.bfloat16,
):
    if skip_load_from_pretrain:
        return WanVideoVAE38().to(device=device, dtype=torch_dtype)
    if checkpoint_root is None:
        raise ValueError(
            '`checkpoint_root` is required for WanVideoVAE38 unless '
            '`skip_load_from_pretrain=True`.')
    root = _resolve_checkpoint_root(checkpoint_root)
    if checkpoint_pattern is None:
        path = _glob_first_required(
            root, ('Wan2.2_VAE.pth', 'Wan2.2_VAE.safetensors'))
    else:
        path = _glob_required(root, checkpoint_pattern)
    return _load_vae(path, torch_dtype=torch_dtype, device=device)


def build_wan_video_dit(
    config: dict[str, Any],
    checkpoint_root: str | None = None,
    checkpoint_pattern: str = 'diffusion_pytorch_model*.safetensors',
    skip_load_from_pretrain: bool = False,
    device: str = 'cuda',
    torch_dtype: torch.dtype = torch.bfloat16,
):
    validated_config = _validate_dit_config(config)
    if skip_load_from_pretrain:
        return WanVideoDiT(**validated_config).to(
            device=device, dtype=torch_dtype)
    if checkpoint_root is None:
        raise ValueError(
            '`checkpoint_root` is required for WanVideoDiT unless '
            '`skip_load_from_pretrain=True`.')
    root = _resolve_checkpoint_root(checkpoint_root)
    path = _glob_required(root, checkpoint_pattern)
    return _load_dit(
        path,
        dit_config=validated_config,
        torch_dtype=torch_dtype,
        device=device,
    )


def build_action_dit(
    config: dict[str, Any],
    pretrained_path: str | None = None,
    skip_load_from_pretrain: bool = False,
    device: str = 'cuda',
    torch_dtype: torch.dtype = torch.bfloat16,
):
    return ActionDiT.from_pretrained(
        action_dit_config=config,
        action_dit_pretrained_path=pretrained_path,
        skip_dit_load_from_pretrain=skip_load_from_pretrain,
        device=device,
        torch_dtype=torch_dtype,
    )
