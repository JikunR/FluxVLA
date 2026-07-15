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

from functools import partial
from typing import Any, Callable, Dict, Optional

import timm
import torch
from timm.models.vision_transformer import Block, VisionTransformer
from torch.distributed.fsdp.wrap import (_module_wrap_policy, _or_policy,
                                         transformer_auto_wrap_policy)

from fluxvla.engines import VISION_BACKBONES
from .base_vision import VisionBackbone
from .configs import VISION_BACKBONE_CONFIGS


@VISION_BACKBONES.register_module()
class DinoViTBackbone(VisionBackbone):
    """DINO ViT backbone for already-preprocessed pixel values."""

    def __init__(
        self,
        vision_backbone_id: str = 'dino',
        model_id: Optional[str] = None,
        pretrained: bool = True,
        pretrained_cfg: Optional[Dict[str, Any]] = None,
        img_size: int = 224,
        feature_layer_offset: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(vision_backbone_id)
        self.feature_layer_offset = int(feature_layer_offset)

        if model_id is None:
            model_id = VISION_BACKBONE_CONFIGS[vision_backbone_id]['model_id']
        create_kwargs = dict(
            pretrained=bool(pretrained),
            num_classes=0,
            img_size=img_size,
        )
        if pretrained_cfg is not None:
            create_kwargs['pretrained_cfg'] = dict(pretrained_cfg)
        create_kwargs.update(kwargs)
        self.vision: VisionTransformer = timm.create_model(
            model_id, **create_kwargs)

    @property
    def embed_dim(self) -> int:
        return int(self.vision.embed_dim)

    @property
    def output_dim(self) -> int:
        return self.embed_dim

    @property
    def num_patches(self) -> int:
        return int(self.vision.patch_embed.num_patches)

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16

    def get_fsdp_wrapping_policy(self) -> Callable:
        vit_wrap_policy = partial(
            _module_wrap_policy, module_classes={VisionTransformer})
        transformer_block_policy = partial(
            transformer_auto_wrap_policy, transformer_layer_cls={Block})
        return partial(
            _or_policy, policies=[vit_wrap_policy, transformer_block_policy])

    def _extract_patch_features(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        if hasattr(self.vision, 'get_intermediate_layers'):
            num_blocks = len(getattr(self.vision, 'blocks'))
            layer_idx = max(0, num_blocks - self.feature_layer_offset)
            features = self.vision.get_intermediate_layers(
                pixel_values,
                n={layer_idx},
                reshape=False,
                return_prefix_tokens=False,
                norm=False,
            )
            return features[0] if isinstance(features, tuple) else features
        return self.vision(pixel_values)

    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        if pixel_values.ndim != 4 or pixel_values.shape[1] != 3:
            raise ValueError(
                '`pixel_values` must be [B,3,H,W] for DinoViTBackbone, '
                f'got shape {tuple(pixel_values.shape)}.')
        return self._extract_patch_features(pixel_values)
