# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations
import copy
import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from fluxvla.engines import (build_projector_from_cfg,
                             build_tokenizer_from_cfg,
                             build_vlm_backbone_from_cfg)
from fluxvla.models.projectors.domain_aware_linear import DomainAwareLinear
from fluxvla.tokenizers.cosmos3_wan22_vae import Cosmos3Wan22VAE


class Cosmos3ComponentsMixin:

    def _build_vlm_backbone(
        self,
        vlm_backbone: Dict,
    ) -> nn.Module:
        if vlm_backbone is None:
            raise ValueError(
                'Cosmos3FlowMatching requires `vlm_backbone` config.')
        cfg = copy.deepcopy(vlm_backbone)
        if 'vlm_config' not in cfg:
            raise ValueError(
                'Cosmos3FlowMatching requires `vlm_backbone.vlm_config`.')
        cfg.setdefault('packed_attention_backend',
                       self.packed_attention_backend)
        return build_vlm_backbone_from_cfg(cfg)

    @staticmethod
    def _build_projector(
        projector: Optional[Dict | nn.Module],
        default_cfg: Dict,
    ) -> nn.Module:
        if isinstance(projector, nn.Module):
            return projector
        cfg = copy.deepcopy(default_cfg if projector is None else projector)
        for key, value in default_cfg.items():
            cfg.setdefault(key, value)
        return build_projector_from_cfg(cfg)

    def _build_vision_vae(
        self,
        vision_vae: Dict | Cosmos3Wan22VAE,
    ) -> Cosmos3Wan22VAE:
        if isinstance(vision_vae, dict):
            cfg = copy.deepcopy(vision_vae)
            cfg.setdefault('type', 'Cosmos3Wan22VAE')
            return build_tokenizer_from_cfg(
                cfg,
                default_args=dict(dtype=self.torch_dtype or torch.bfloat16),
            )
        return vision_vae

    @staticmethod
    def _first_parameter(module: nn.Module) -> torch.nn.Parameter:
        return next(module.parameters())

    @staticmethod
    def _init_linear_like_cosmos3(module: nn.Module, in_dim: int) -> None:
        linear = getattr(module, 'projector', module)
        if not isinstance(linear, nn.Linear):
            return
        std = 1.0 / math.sqrt(in_dim)
        nn.init.trunc_normal_(linear.weight, std=std, a=-3 * std, b=3 * std)
        if linear.bias is not None:
            nn.init.zeros_(linear.bias)

    @staticmethod
    def _init_domain_aware_like_cosmos3(module: nn.Module) -> None:
        if not isinstance(module, DomainAwareLinear):
            return
        std = 1.0 / math.sqrt(module.input_size)
        nn.init.trunc_normal_(module.fc.weight, std=std, a=-3 * std, b=3 * std)
        nn.init.zeros_(module.bias.weight)

    def _init_projection_weights_like_cosmos3(self) -> None:
        if hasattr(self.time_embedder, '_init_weights'):
            self.time_embedder._init_weights()
        self._init_linear_like_cosmos3(self.vision_in_proj,
                                       self.patch_latent_dim)
        self._init_linear_like_cosmos3(self.vision_out_proj, self.hidden_size)
        for projector in (self.action_in_proj, self.action_out_proj):
            self._init_domain_aware_like_cosmos3(projector)
        std = 1.0 / math.sqrt(self.hidden_size)
        nn.init.trunc_normal_(
            self.action_modality_embed, std=std, a=-3 * std, b=3 * std)

    @staticmethod
    def _projector_dim(module: nn.Module, kind: str) -> Optional[int]:
        attr_name = 'input_size' if kind == 'in' else 'output_size'
        if hasattr(module, attr_name):
            return int(getattr(module, attr_name))
        attr_name = 'in_dim' if kind == 'in' else 'out_dim'
        if hasattr(module, attr_name):
            return int(getattr(module, attr_name))
        linear = getattr(module, 'projector', module)
        if isinstance(linear, nn.Linear):
            return int(linear.in_features if kind ==
                       'in' else linear.out_features)
        return None

    def _validate_projector_shape(
        self,
        name: str,
        module: nn.Module,
        *,
        in_dim: int,
        out_dim: int,
        num_embodiments: Optional[int] = None,
    ) -> None:
        actual_in_dim = self._projector_dim(module, 'in')
        actual_out_dim = self._projector_dim(module, 'out')
        errors = []
        if actual_in_dim is not None and actual_in_dim != in_dim:
            errors.append(f'in_dim={actual_in_dim}, expected {in_dim}')
        if actual_out_dim is not None and actual_out_dim != out_dim:
            errors.append(f'out_dim={actual_out_dim}, expected {out_dim}')
        if num_embodiments is not None:
            actual_num = getattr(module, 'num_embodiments',
                                 getattr(module, 'num_domains', None))
            if actual_num is not None and int(actual_num) != num_embodiments:
                errors.append(f'num_embodiments={actual_num}, expected '
                              f'{num_embodiments}')
        if errors:
            raise ValueError(f'Cosmos3FlowMatching {name} shape mismatch: '
                             f'{", ".join(errors)}.')

    def _validate_projector_shapes(self) -> None:
        self._validate_projector_shape(
            'vision_in_proj',
            self.vision_in_proj,
            in_dim=self.patch_latent_dim,
            out_dim=self.hidden_size,
        )
        self._validate_projector_shape(
            'vision_out_proj',
            self.vision_out_proj,
            in_dim=self.hidden_size,
            out_dim=self.patch_latent_dim,
        )
        self._validate_projector_shape(
            'action_in_proj',
            self.action_in_proj,
            in_dim=self.max_action_dim,
            out_dim=self.hidden_size,
            num_embodiments=self.num_embodiment_domains,
        )
        self._validate_projector_shape(
            'action_out_proj',
            self.action_out_proj,
            in_dim=self.hidden_size,
            out_dim=self.max_action_dim,
            num_embodiments=self.num_embodiment_domains,
        )

    @staticmethod
    def _read_config_value(config, name: str):
        if isinstance(config, dict):
            return config[name]
        return getattr(config, name)

    def _vlm_text_config(self):
        text_config = getattr(self.vlm_backbone, 'text_config', None)
        if text_config is not None:
            return text_config
        model = getattr(self.vlm_backbone, 'model', None)
        if model is not None and hasattr(model, 'config'):
            return model.config
        raise ValueError('Cosmos3FlowMatching requires vlm_backbone to expose '
                         '`text_config` or `model.config`.')

    def _text_pad_token_id(self) -> Optional[int]:
        text_config = self._vlm_text_config()
        if isinstance(text_config, dict):
            value = text_config.get('pad_token_id')
        else:
            value = getattr(text_config, 'pad_token_id', None)
        return None if value is None else int(value)

    def _derive_hidden_size_from_vlm_config(self) -> int:
        text_config = self._vlm_text_config()
        return int(self._read_config_value(text_config, 'hidden_size'))

    def _keep_rotary_buffers_fp32(self) -> None:
        for module in self.modules():
            for buffer_name in ('inv_freq', 'original_inv_freq'):
                buffer = getattr(module, buffer_name, None)
                if isinstance(buffer,
                              torch.Tensor) and buffer.is_floating_point():
                    setattr(module, buffer_name, buffer.float())

    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        self.vision_vae.to(*args, **kwargs)
        self.vision_vae.eval().requires_grad_(False)
        self._keep_rotary_buffers_fp32()
        return module
