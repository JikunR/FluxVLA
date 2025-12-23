# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa: E501
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import partial
from typing import Callable, Dict, List, Optional, Type

import torch
from torch import nn
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoConfig, AutoModel
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen3.modeling_qwen3 import (Qwen3Attention,
                                                      Qwen3DecoderLayer,
                                                      Qwen3MLP)

from fluxvla.engines import VLM_BACKBONES, str_to_dtype


@VLM_BACKBONES.register_module()
class EagleBackbone(nn.Module):
    """
    HuggingFace-compatible wrapper for Eagle VLMs.
    Inherits from VLMBackbone and provides Eagle-specific token
    and configuration handling. This includes explicitly adding a padding
    token to the tokenizer and resizing the model embeddings accordingly.
    Registered into the `VLM_BACKBONES` registry for flexible instantiation.

    Args:
        vlm_backbone_id (str): Identifier string for this backbone.
        vlm_config (Dict, optional): Configuration dictionary for the
            VLM.
        vlm_path (Optional[str]): Path to the VLM model weights.
            If None, the model will be loaded from the HuggingFace hub.
            Defaults to None.
    """

    def __init__(
        self,
        vlm_path: str,
        vlm_config: Dict = None,
        project_to_dim: Optional[int] = None,
        select_layer: int = 12,
        dtype='float32',
    ):
        super().__init__()

        config = AutoConfig.from_pretrained(vlm_path, trust_remote_code=True)
        if dtype is not None:
            self.vlm = AutoModel.from_config(
                config, trust_remote_code=True).to(str_to_dtype(dtype))
        else:
            self.vlm = AutoModel.from_config(config, trust_remote_code=True)

        if project_to_dim is not None:
            self.eagle_linear = torch.nn.Linear(2048, project_to_dim)
        else:
            self.eagle_linear = torch.nn.Identity()

        # needed since we don't use these layers. Also saves compute
        while len(self.vlm.language_model.model.layers) > select_layer:
            self.vlm.language_model.model.layers.pop(-1)

        self.select_layer = select_layer
        self.config = config

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return Qwen3DecoderLayer

    def forward_eagle(self, vl_input: BatchFeature) -> BatchFeature:
        eagle_prefix = 'eagle_'
        eagle_input = {
            k.removeprefix(eagle_prefix): v
            for k, v in vl_input.items() if k.startswith(eagle_prefix)
        }
        del eagle_input['image_sizes']

        eagle_output = self.vlm(
            **eagle_input, output_hidden_states=True, return_dict=True)
        eagle_features = eagle_output.hidden_states[self.select_layer]

        eagle_features = self.eagle_linear(eagle_features)
        return eagle_features, eagle_input['attention_mask']

    def forward(self,
                images: List[torch.Tensor],
                lang_tokens: torch.Tensor,
                img_masks: torch.Tensor,
                lang_masks: Optional[torch.Tensor] = None,
                *args,
                **kwargs) -> BatchFeature:

        vlm_output = self.vlm(
            input_ids=lang_tokens,
            attention_mask=lang_masks,
            pixel_values=images.reshape(
                (images.shape[0] * images.shape[1] // 3, 3, images.shape[2],
                 images.shape[3])),
            output_hidden_states=True,
            return_dict=True)

        # YL (TODO HACK): to resolve DDP issue when tune_visual=True
        # Ensure all trainable parameters in vision_model are used
        # in the forward pass for DDP compatibility
        eagle_features = vlm_output.hidden_states[self.select_layer]

        eagle_features = self.eagle_linear(eagle_features)
        return eagle_features, lang_masks, None

    def enable_gradient_checkpointing(self) -> None:
        """
        Enables gradient checkpointing on the underlying Qwen3 LLM model.
        Uses HuggingFace's built-in gradient checkpointing mechanism.
        """
        # Use use_reentrant=False for better compatibility with FSDP
        gradient_checkpointing_kwargs = {'use_reentrant': False}
        if hasattr(self.vlm, 'language_model'):
            self.vlm.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        elif hasattr(self.vlm, 'gradient_checkpointing_enable'):
            self.vlm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def get_fsdp_wrapping_policy(self) -> Callable:
        """
        Returns a function used to determine which modules to wrap with FSDP.

        Returns:
            Callable: Wrapping policy function.
        """
        transformer_block_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Qwen3Attention, Qwen3MLP},
        )
        return transformer_block_policy
