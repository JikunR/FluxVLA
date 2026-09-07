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

from collections.abc import Mapping
from typing import Any, Optional

import torch
import torch.nn as nn

from fluxvla.engines import VLM_BACKBONES
from fluxvla.engines.utils.name_map import str_to_dtype
from .wan22_loader import build_wan_text_encoder

__all__ = ['Wan22TextBackbone']


@VLM_BACKBONES.register_module()
class Wan22TextBackbone(nn.Module):
    """Wan2.2 text encoder exposed as a regular VLM backbone.

    This keeps the WAM-specific VAE outside ``vlm_backbone`` while still
    allowing the current Wan UMT5 context path to be configured through the
    same top-level backbone slot used by other VLA models.
    """

    def __init__(
        self,
        checkpoint_root: Optional[str] = None,
        checkpoint_pattern: Optional[str] = None,
        context_encoder: Optional[Mapping[str, Any] | nn.Module] = None,
        device: str = 'cpu',
        torch_dtype: torch.dtype = torch.float32,
        skip_load: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(torch_dtype, str):
            torch_dtype = str_to_dtype(torch_dtype)
        self._device = torch.device(device)
        self.torch_dtype = torch_dtype

        self.context_encoder = self._build_context_encoder(
            context_encoder,
            checkpoint_root=checkpoint_root,
            checkpoint_pattern=checkpoint_pattern,
            device=str(self._device),
            torch_dtype=self.torch_dtype,
            skip_load=skip_load,
        )
        self.requires_grad_(False)

    @staticmethod
    def _build_context_encoder(
        context_encoder: Optional[Mapping[str, Any] | nn.Module],
        *,
        checkpoint_root: Optional[str],
        checkpoint_pattern: Optional[str],
        device: str,
        torch_dtype: torch.dtype,
        skip_load: bool,
    ):
        if context_encoder is None:
            return build_wan_text_encoder(
                checkpoint_root=checkpoint_root,
                checkpoint_pattern=checkpoint_pattern,
                device=device,
                torch_dtype=torch_dtype,
                skip_load_from_pretrain=skip_load,
            )
        if isinstance(context_encoder, nn.Module):
            return context_encoder
        if not isinstance(context_encoder, Mapping):
            raise TypeError(
                '`vlm_backbone.context_encoder` must be a dict or nn.Module, '
                f'got {type(context_encoder).__name__}.')

        cfg = dict(context_encoder)
        builder = cfg.pop('type', None)
        if not callable(builder):
            raise TypeError(
                '`vlm_backbone.context_encoder.type` must be a callable when '
                f'provided, got {type(builder).__name__}.')

        cfg.setdefault('device', device)
        cfg.setdefault('torch_dtype', torch_dtype)
        if skip_load:
            cfg['skip_load_from_pretrain'] = True
        return builder(**cfg)

    @property
    def device(self) -> torch.device:
        for tensor in self.parameters():
            return tensor.device
        for tensor in self.buffers():
            return tensor.device
        return self._device

    def set_frozen_modules_to_eval_mode(self) -> None:
        self.context_encoder.eval()

    @torch.no_grad()
    def forward(self,
                images=None,
                lang_tokens=None,
                img_masks=None,
                lang_masks=None,
                *args,
                **kwargs):
        del images, img_masks, args, kwargs
        if lang_tokens is None or lang_masks is None:
            raise ValueError('`Wan22TextBackbone.forward` requires '
                             '`lang_tokens/lang_masks`.')
        ids = lang_tokens.to(self.device)
        mask = lang_masks.to(self.device, dtype=torch.bool)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        prompt_emb = self.context_encoder(ids, mask)
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        return prompt_emb.to(device=self.device), torch.ones_like(mask)
