# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# flake8: noqa

from __future__ import annotations
from typing import List, Optional

import torch

from fluxvla.models.third_party_models.cosmos3.data.vfm.sequence_packing import (
    GenerationDataClean, PackedSequence, SequencePlan, pack_input_sequence)
from .flow_utils import _move_packed_sequence_to_device


class Cosmos3SequenceMixin:

    def _pack_generation_data(
        self,
        *,
        sequence_plans: List[SequencePlan],
        text_token_ids: List[List[int]],
        vision_tokens: Optional[List[torch.Tensor]],
        action_tokens: Optional[List[torch.Tensor]],
        embodiment_ids: Optional[List[torch.Tensor]],
        raw_action_dim: Optional[List[torch.Tensor]],
        timesteps: torch.Tensor,
        fps_vision: Optional[torch.Tensor],
        fps_action: Optional[torch.Tensor],
    ) -> PackedSequence:
        gen_data_clean = GenerationDataClean(
            batch_size=len(sequence_plans),
            is_image_batch=False,
            x0_tokens_vision=vision_tokens,
            fps_vision=fps_vision,
            x0_tokens_action=action_tokens,
            fps_action=fps_action,
            action_domain_id=embodiment_ids,
            raw_action_dim=raw_action_dim,
        )
        packed_seq = pack_input_sequence(
            sequence_plans=sequence_plans,
            input_text_indexes=text_token_ids,
            gen_data_clean=gen_data_clean,
            input_timesteps=timesteps.detach().cpu(),
            special_tokens=self.special_tokens,
            latent_patch_size=self.latent_patch_size,
            include_end_of_generation_token=False,
            position_embedding_type=self.position_embedding_type,
            unified_3d_mrope_reset_spatial_ids=self.
            unified_3d_mrope_reset_spatial_ids,
            unified_3d_mrope_temporal_modality_margin=self.
            unified_3d_mrope_temporal_modality_margin,
            enable_fps_modulation=self.enable_fps_modulation,
            base_fps=self.base_fps,
            temporal_compression_factor=(
                self.vision_vae.temporal_compression_factor),
            action_dim=self.max_action_dim,
        )
        return _move_packed_sequence_to_device(packed_seq, self.device)
