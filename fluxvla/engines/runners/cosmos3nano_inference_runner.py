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
"""Inference runner and inference dataset for Cosmos3-Nano VLA.

Follows the same pattern as ``AlohaInferenceRunner`` /
``PrivateInferenceDataset``, but builds inputs for ``Cosmos3NanoVLA``.

Weight loading differs from other VLAs because Cosmos3NanoVLA splits
initialization into two steps:

1. ``build_vla_from_cfg`` → ``__init__``: builds tokenizer + schedulers only;
   ``net`` and ``vae`` are ``None``.
2. ``vla.from_pretrained()``: loads the HF backbone (``net``) and VAE.
3. *(optional)* load a FluxVLA finetune checkpoint on top of ``net``.

``Cosmos3NanoInferenceRunner.run_setup`` handles all three steps.
"""

from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from fluxvla.engines.utils import (build_transform_from_cfg,
                                   initialize_overwatch)
from fluxvla.engines.utils.name_map import str_to_dtype
from fluxvla.engines.utils.root import DATASETS, RUNNERS
from .base_inference_runner import BaseInferenceRunner

overwatch = initialize_overwatch(__name__)

# ---------------------------------------------------------------------------
# Cosmos3NanoInferenceDataset
# ---------------------------------------------------------------------------


@DATASETS.register_module()
class Cosmos3NanoInferenceDataset:
    """Preprocess a single real-robot observation for ``Cosmos3NanoVLA``.

    Applies the same transform pipeline as training (resize → normalize →
    build_sequence), then assembles a ready-to-run batch dict with correct
    video shape ``[1, C, T, H_tiled, W]``.

    Args:
        norm_stats (str | dict): Path to a JSON file containing normalization
            statistics, or the statistics dict directly.
        transforms (List[Dict]): Transform pipeline configs
            (ResizeImages, SimpleNormalizeImages, NormalizeStatesAndActions).
            Do NOT include PrepareVideo here — video assembly is done internally  # noqa: E501
            to match inference's single-frame input layout.
        qwen3_vl_model_path (str): Path to the Qwen3-VL checkpoint (for the
            ``ProcessCosmos3NanoPrompt`` tokenizer).
        img_keys (List[str]): Image field names in the incoming observation.
        embodiment_id (int): Integer embodiment ID used in training.
        domain_id (int): Cosmos3 domain ID (e.g. 8 for droid_lerobot).
        max_action_dim (int): Padded action dimension.
        action_horizon (int): Number of action steps to predict.
        frame_window_size (int): Number of video frames per window.
            When ``frame_window_size > 1``, the runner is expected to pass a
            stack of historical frames; for online deployment with a single
            current frame pass ``frame_window_size=1``.
        num_views (int): Number of camera views tiled vertically.
    """

    def __init__(
        self,
        norm_stats: str | dict,
        transforms: List[Dict],
        qwen3_vl_model_path: str = '',
        img_keys: Optional[List[str]] = None,
        embodiment_id: int = 0,
        domain_id: int = 8,
        max_action_dim: int = 64,
        action_horizon: int = 16,
        frame_window_size: int = 1,
        num_views: int = 2,
    ) -> None:
        self.img_keys = img_keys or ['head', 'left_wrist']
        self.embodiment_id = embodiment_id
        self.domain_id = domain_id
        self.max_action_dim = max_action_dim
        self.action_horizon = action_horizon
        self.frame_window_size = frame_window_size
        self.num_views = num_views

        # Load norm stats
        if isinstance(norm_stats, str) and norm_stats:
            with open(norm_stats, 'r') as f:
                self.norm_stats = json.load(f)
        elif isinstance(norm_stats, dict):
            self.norm_stats = norm_stats
        else:
            self.norm_stats = {}

        # Build transform pipeline — inject qwen3_vl_model_path where needed
        self.transforms: list = []
        for t_cfg in transforms:
            if t_cfg.get(
                    'type'
            ) == 'ProcessCosmos3NanoPrompt' and qwen3_vl_model_path:
                t_cfg = dict(t_cfg)
                t_cfg.setdefault('qwen3_vl_model_path', qwen3_vl_model_path)
            self.transforms.append(build_transform_from_cfg(t_cfg))

    def __call__(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a raw robot observation to a batch dict.

        Args:
            obs: Dict with:
                - image keys (HWC uint8 numpy arrays, one per ``img_keys``)
                  OR ``'images'`` key containing a list/array already
                - ``'qpos'``: joint positions numpy array
                - ``'task_description'``: str

        Returns:
            Dict suitable for ``Cosmos3NanoVLA.predict_action()``.
            All tensors have a leading batch dimension of 1 and are on CUDA.
        """
        # ---- collect raw images per view ---------------------------------
        if 'images' in obs:
            # Already assembled by caller
            raw_imgs = obs['images']
            if isinstance(raw_imgs, np.ndarray):
                raw_imgs = list(raw_imgs)
        else:
            raw_imgs = []
            for key in self.img_keys:
                if key not in obs:
                    raise KeyError(
                        f'Cosmos3NanoInferenceDataset: missing image key {key!r}'  # noqa: E501
                    )
                img = obs[key]
                if img.ndim == 3 and img.shape[-1] == 3:
                    img = img.transpose(2, 0, 1)  # HWC → CHW
                raw_imgs.append(img)

        inputs: Dict = {
            'images': raw_imgs,  # list of CHW arrays, one per view
            'task_description': obs.get('task_description', ''),
            'stats': self.norm_stats.get('private', self.norm_stats),
            'states': obs.get('qpos', np.zeros(64, dtype=np.float32)),
            'embodiment_ids': np.array(self.embodiment_id, dtype=np.int64),
        }

        # Run configured transforms (resize / normalize / tokenize / etc.)
        for t in self.transforms:
            inputs = t(inputs)

        # ---- assemble video tensor [C, T, H_tiled, W] -------------------
        # Inference typically uses a single frame (T=1).  Multiple views are
        # tiled vertically (same layout as PrepareVideo in training).
        imgs = inputs['images']
        if isinstance(imgs, np.ndarray):
            imgs = [imgs] if imgs.ndim == 3 else list(imgs)
        # Each element: CHW float32 in [-1, 1] after SimpleNormalizeImages
        # Stack views vertically → [C, H*V, W], then add T dim → [C, T, H*V, W]
        stacked_views = np.concatenate(imgs, axis=1)  # [C, H*num_views, W]
        video_np = stacked_views[:, np.newaxis, :, :]  # [C, 1, H*V, W]

        # ---- build batch with leading B=1 dim and move to CUDA -----------
        def _to_tensor_cuda(x, dtype=torch.float32):
            if isinstance(x, np.ndarray):
                t = torch.from_numpy(x)
            else:
                t = x
            return t.unsqueeze(0).to(device='cuda', dtype=dtype)

        batch = {
            # [1, C, T, H_tiled, W]
            'images':
            _to_tensor_cuda(video_np),
            # [1, L]
            'text_token_ids':
            _to_tensor_cuda(inputs['text_token_ids'], dtype=torch.long),
            # [1]
            'domain_id':
            torch.tensor([self.domain_id], dtype=torch.long, device='cuda'),
            # [1]
            'raw_action_dim':
            torch.tensor(
                [int(inputs.get('raw_action_dim', self.max_action_dim))],
                dtype=torch.long,
                device='cuda',
            ),
            'action_horizon':
            self.action_horizon,
        }
        return batch


# ---------------------------------------------------------------------------
# Cosmos3NanoInferenceRunner
# ---------------------------------------------------------------------------


@RUNNERS.register_module()
class Cosmos3NanoInferenceRunner(BaseInferenceRunner):
    """Real-robot inference runner for Cosmos3-Nano VLA.

    Follows the ``AlohaInferenceRunner`` structure:
    observation → preprocessing → ``predict_action`` → denormalize → execute.

    Weight loading sequence (see module docstring):
    ``__init__`` → ``run_setup`` → backbone loaded by ``vla.from_pretrained()``
    → optional finetune checkpoint overlaid on ``vla.net``.

    Args:
        action_dim (int): True (unpadded) action dimension for the robot.
        action_horizon (int): Number of action timesteps to execute per chunk.
        All other args are passed to ``BaseInferenceRunner``.
    """

    def __init__(
        self,
        action_dim: int = 52,
        action_horizon: int = 16,
        *args,
        **kwargs,
    ) -> None:
        self.action_dim = action_dim
        self.action_horizon = action_horizon

        # Pass action_chunk so BaseInferenceRunner slices correctly.
        kwargs.setdefault('action_chunk', action_horizon)

        # Cosmos3NanoVLA does NOT go through the BaseInferenceRunner
        # ckpt_path + load_state_dict path (net is None at __init__ time).
        # Stash ckpt_path before calling super().__init__ so we can handle
        # it ourselves in run_setup().
        self._cosmos3_ckpt_path = kwargs.pop('ckpt_path', None)

        # Build the VLA structure (tokenizer + schedulers only — no weights)
        # via BaseInferenceRunner, but tell it ckpt_path=None so it won't
        # attempt an early load_state_dict.
        super().__init__(*args, ckpt_path=None, **kwargs)

    # ------------------------------------------------------------------
    # Override run_setup to perform the two-step weight loading
    # ------------------------------------------------------------------

    def run_setup(self) -> None:
        """Load backbone + VAE, optionally overlay finetune checkpoint,
        then configure the model for inference (eval, dtype cast)."""
        import torch

        # Step 1: load HF backbone + frozen VAE
        overwatch.info(
            '[Cosmos3NanoInferenceRunner] Loading backbone weights…')
        self.vla.from_pretrained()

        # Step 2: overlay a FluxVLA finetune checkpoint if provided
        if self._cosmos3_ckpt_path is not None:
            overwatch.info(
                f'[Cosmos3NanoInferenceRunner] Loading finetune checkpoint: '
                f'{self._cosmos3_ckpt_path}')
            from safetensors.torch import load_file as st_load
            ckpt = self._cosmos3_ckpt_path
            if ckpt.endswith('.safetensors'):
                state_dict = st_load(ckpt, device='cpu')
            elif os.path.isdir(ckpt):
                state_dict = {}
                for fname in os.listdir(ckpt):
                    if fname.endswith('.safetensors'):
                        state_dict.update(
                            st_load(os.path.join(ckpt, fname), device='cpu'))
            else:
                import torch as _torch
                ckpt_data = _torch.load(ckpt, map_location='cpu')
                state_dict = ckpt_data.get('model', ckpt_data)

            # Strip 'net.' prefix and load into self.vla.net only
            net_weights = {}
            for k, v in state_dict.items():
                if k.startswith('net.'):
                    net_weights[k[len('net.'):]] = v
            if net_weights:
                missing, unexpected = self.vla.net.load_state_dict(
                    net_weights, strict=False)
                if missing:
                    overwatch.warning(
                        f'[Cosmos3NanoInferenceRunner] {len(missing)} missing '
                        f'keys in finetune checkpoint')
                overwatch.info(
                    f'[Cosmos3NanoInferenceRunner] Finetune checkpoint loaded '
                    f'({len(net_weights)} tensors)')

        # Step 3: configure for inference
        dtype = str_to_dtype(self.mixed_precision_dtype) \
            if hasattr(self.mixed_precision_dtype, '__str__') \
            else self.mixed_precision_dtype
        self.vla.net.eval().to(device='cuda', dtype=dtype)

        # VAE uses bfloat16 internally; just move to CUDA
        if hasattr(self.vla.vae, 'model') and hasattr(self.vla.vae.model,
                                                      'cuda'):
            self.vla.vae.model.cuda()

        overwatch.info('[Cosmos3NanoInferenceRunner] Model ready.')

    # ------------------------------------------------------------------
    # BaseInferenceRunner hooks
    # ------------------------------------------------------------------

    def _predict_action(self, inputs: Dict) -> np.ndarray:
        """Run ``Cosmos3NanoVLA.predict_action`` and return raw numpy actions."""  # noqa: E501
        action_horizon = inputs.pop('action_horizon', self.action_horizon)
        with torch.inference_mode():
            actions = self.vla.predict_action(
                **inputs,
                action_horizon=action_horizon,
            )
        # actions: [1, T_act, D] → [T_act, D] numpy
        return actions[0].cpu().float().numpy()

    def _postprocess_actions(self, raw_action: np.ndarray) -> np.ndarray:
        """Denormalize actions from normalized space to robot command space.

        Args:
            raw_action: ``[T_act, action_dim]`` float32 numpy array.
        """
        if hasattr(
                self,
                'denormalize_action') and self.denormalize_action is not None:
            result = self.denormalize_action(
                {'action': raw_action[:, :self.action_dim]})
            return result['action']
        return raw_action[:, :self.action_dim]
