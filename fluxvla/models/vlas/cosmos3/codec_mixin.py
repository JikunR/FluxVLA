# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# flake8: noqa

from __future__ import annotations
from typing import List, Optional, Tuple

import torch

from fluxvla.models.backbones.vlms.cosmos3.cosmos3_attention import \
    build_packed_sequence
from fluxvla.models.third_party_models.cosmos3.data.vfm.sequence_packing import (
    PackedSequence, get_all_seq)
from .flow_utils import (_apply_timestep_embeds_to_noisy_tokens,
                         _as_list_of_1chw)


class Cosmos3CodecMixin:
    """Encode and decode text, Wan latents, and action tokens.

    These helpers sit between the Cosmos3 sequence packer and the MoT backbone:
    they project concrete modality tensors into hidden states, run the packed
    backbone, and project noisy prediction slots back to modality space.
    """

    def _tokenize_vision(
            self,
            images: Optional[torch.Tensor]) -> Optional[List[torch.Tensor]]:
        if images is None:
            return None
        vae_images = images.to(
            device=self.vision_vae.device, dtype=self.vision_vae.dtype)
        latents = self.vision_vae.encode(vae_images)
        model_dtype = self._first_parameter(self.vision_in_proj).dtype
        latents = latents.to(device=self.device, dtype=model_dtype)
        return [latents[i:i + 1] for i in range(latents.shape[0])]

    def patchify_and_pack_latents(
        self,
        tokens_vision: List[torch.Tensor],
        token_shapes_vision: List[Tuple[int, int, int]],
    ) -> tuple[torch.Tensor, List[Tuple[int, int, int]]]:
        packed_latent = []
        original_latent_shapes = []
        p = self.latent_patch_size

        for latent, _ in zip(tokens_vision, token_shapes_vision):
            latent = latent.squeeze(0)
            _, t_actual, h_actual, w_actual = latent.shape
            original_latent_shapes.append((t_actual, h_actual, w_actual))
            h_padded = ((h_actual + p - 1) // p) * p
            w_padded = ((w_actual + p - 1) // p) * p
            if h_padded != h_actual or w_padded != w_actual:
                padded = latent.new_zeros(
                    (self.latent_channel, t_actual, h_padded, w_padded))
                padded[:, :, :h_actual, :w_actual] = latent
                latent = padded

            h_patches = h_padded // p
            w_patches = w_padded // p
            latent = latent.reshape(self.latent_channel, t_actual, h_patches,
                                    p, w_patches, p)
            latent = torch.einsum('cthpwq->thwpqc',
                                  latent).reshape(-1, self.patch_latent_dim)
            packed_latent.append(latent)

        return torch.cat(packed_latent, dim=0), original_latent_shapes

    def unpatchify_and_unpack_latents(
        self,
        packed_mse_preds: torch.Tensor,
        token_shapes_vision: List[Tuple[int, int, int]],
        noisy_frame_indexes_vision: List[torch.Tensor],
        original_latent_shapes: Optional[List[Tuple[int, int, int]]] = None,
    ) -> List[torch.Tensor]:
        p = self.latent_patch_size
        unpatchified_latents = []
        start_idx = 0

        for index, (t_c, h_c, w_c) in enumerate(token_shapes_vision):
            if original_latent_shapes is not None:
                _, h_orig, w_orig = original_latent_shapes[index]
                h_patches = ((h_orig + p - 1) // p)
                w_patches = ((w_orig + p - 1) // p)
            else:
                h_orig, w_orig = h_c * p, w_c * p
                h_patches, w_patches = h_c, w_c

            noisy_frame_indexes = noisy_frame_indexes_vision[index].to(
                device=packed_mse_preds.device, dtype=torch.long)
            output_tensor = packed_mse_preds.new_zeros(
                (self.latent_channel, t_c, h_orig, w_orig))
            num_patches = len(noisy_frame_indexes) * h_patches * w_patches
            if num_patches > 0:
                end_idx = start_idx + num_patches
                latent_patches = packed_mse_preds[start_idx:end_idx]
                latent = latent_patches.reshape(
                    len(noisy_frame_indexes),
                    h_patches,
                    w_patches,
                    p,
                    p,
                    self.latent_channel,
                )
                latent = torch.einsum('thwpqc->cthpwq', latent)
                latent = latent.reshape(self.latent_channel,
                                        len(noisy_frame_indexes),
                                        h_patches * p, w_patches * p)
                output_tensor[:, noisy_frame_indexes] = latent[:, :, :h_orig, :
                                                               w_orig]
                start_idx = end_idx
            unpatchified_latents.append(output_tensor.unsqueeze(0))
        return unpatchified_latents

    def pack_action(
        self,
        tokens_action: List[torch.Tensor],
        token_shapes_action: List[Tuple[int, ...]],
        embodiment_ids_action: List[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        packed = []
        embodiment_ids = []
        for tokens, shape, embodiment_id in zip(tokens_action,
                                                token_shapes_action,
                                                embodiment_ids_action):
            t_steps = shape[0]
            packed.append(tokens[:t_steps])
            embodiment_ids.append(embodiment_id.expand(t_steps))
        return torch.cat(packed, dim=0), torch.cat(embodiment_ids, dim=0)

    def unpack_action(
        self,
        packed_action_preds: torch.Tensor,
        token_shapes_action: List[Tuple[int, ...]],
        noisy_frame_indexes_action: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        unpacked = []
        start_idx = 0
        for shape, noisy_frame_indexes in zip(token_shapes_action,
                                              noisy_frame_indexes_action):
            t_steps = shape[0]
            output = packed_action_preds.new_zeros(
                (t_steps, self.max_action_dim))
            noisy_frame_indexes = noisy_frame_indexes.to(
                device=packed_action_preds.device, dtype=torch.long)
            if len(noisy_frame_indexes) > 0:
                end_idx = start_idx + len(noisy_frame_indexes)
                output[noisy_frame_indexes] = packed_action_preds[
                    start_idx:end_idx]
                start_idx = end_idx
            unpacked.append(output)
        return unpacked

    def _encode_text(
            self,
            packed_seq: PackedSequence) -> tuple[torch.Tensor, torch.dtype]:
        embed_text_ids = getattr(self.vlm_backbone, 'embed_text_ids', None)
        if embed_text_ids is None:
            packed_text_embedding = (
                self.vlm_backbone.model.language_model.embed_tokens(
                    packed_seq.text_ids))
        else:
            packed_text_embedding = embed_text_ids(packed_seq.text_ids)
        packed_sequence = packed_text_embedding.new_zeros(
            (packed_seq.sequence_length, self.hidden_size))
        packed_sequence[packed_seq.text_indexes] = packed_text_embedding
        return packed_sequence, packed_text_embedding.dtype

    def _encode_vision(
        self,
        packed_seq: PackedSequence,
        packed_sequence: torch.Tensor,
        target_dtype: torch.dtype,
    ) -> Optional[List[Tuple[int, int, int]]]:
        vision = packed_seq.vision
        if vision is None or vision.tokens is None:
            return None

        packed_tokens_vision, original_latent_shapes = self.patchify_and_pack_latents(
            vision.tokens,
            vision.token_shapes,
        )
        packed_tokens_vision = self.vision_in_proj(
            packed_tokens_vision.to(
                dtype=self._first_parameter(self.vision_in_proj).dtype)).to(
                    target_dtype)

        if vision.mse_loss_indexes.numel() > 0:
            timesteps_vision = vision.timesteps.to(
                dtype=torch.float32) * self.timestep_scale
            with torch.autocast(
                    'cuda',
                    enabled=timesteps_vision.device.type == 'cuda',
                    dtype=torch.float32,
            ):
                packed_timestep_embeds_vision = self.time_embedder(
                    timesteps_vision)
            packed_timestep_embeds_vision = packed_timestep_embeds_vision.to(
                target_dtype)
            packed_tokens_vision = _apply_timestep_embeds_to_noisy_tokens(
                packed_tokens=packed_tokens_vision,
                packed_timestep_embeds=packed_timestep_embeds_vision,
                noisy_frame_indexes=vision.noisy_frame_indexes,
                token_shapes=vision.token_shapes,
            )

        packed_sequence[vision.sequence_indexes] = packed_tokens_vision
        return original_latent_shapes

    def _encode_action(
        self,
        packed_seq: PackedSequence,
        packed_sequence: torch.Tensor,
        target_dtype: torch.dtype,
    ) -> None:
        action = packed_seq.action
        if action is None or action.tokens is None:
            return

        packed_tokens_action, per_token_embodiment_id = self.pack_action(
            action.tokens,
            action.token_shapes,
            action.domain_id,
        )
        packed_tokens_action = self.action_in_proj(
            packed_tokens_action.to(dtype=self.action_modality_embed.dtype),
            per_token_embodiment_id,
        )
        packed_tokens_action = packed_tokens_action + self.action_modality_embed.view(
            1, -1)
        packed_tokens_action = packed_tokens_action.to(target_dtype)

        if action.mse_loss_indexes.numel() > 0:
            timesteps_action = action.timesteps.to(
                dtype=torch.float32) * self.timestep_scale
            with torch.autocast(
                    'cuda',
                    enabled=timesteps_action.device.type == 'cuda',
                    dtype=torch.float32,
            ):
                packed_timestep_embeds_action = self.time_embedder(
                    timesteps_action)
            packed_timestep_embeds_action = packed_timestep_embeds_action.to(
                target_dtype)
            packed_tokens_action = _apply_timestep_embeds_to_noisy_tokens(
                packed_tokens=packed_tokens_action,
                packed_timestep_embeds=packed_timestep_embeds_action,
                noisy_frame_indexes=action.noisy_frame_indexes,
                token_shapes=action.token_shapes,
            )

        packed_sequence[action.sequence_indexes] = packed_tokens_action

    def _run_backbone(self, packed_seq: PackedSequence,
                      packed_sequence: torch.Tensor) -> torch.Tensor:
        all_gen_indexes = []
        if packed_seq.vision is not None:
            all_gen_indexes.append(packed_seq.vision.sequence_indexes)
        if packed_seq.action is not None:
            all_gen_indexes.append(packed_seq.action.sequence_indexes)
        packed_gen_token_indexes = torch.cat(
            all_gen_indexes,
            dim=0) if all_gen_indexes else packed_seq.text_indexes[:0]

        input_pack, attention_meta = build_packed_sequence(
            packed_sequence=packed_sequence,
            attn_modes=packed_seq.attn_modes,
            split_lens=packed_seq.split_lens,
            sample_lens=packed_seq.sample_lens,
            packed_und_token_indexes=packed_seq.text_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
            is_image_batch=packed_seq.is_image_batch,
            pad_for_cuda_graphs=False,
        )

        def run_vlm_backbone():
            return self.vlm_backbone.forward_packed(
                input_pack,
                attention_mask=attention_meta,
                position_ids=packed_seq.position_ids,
            )

        packed_outputs, _ = run_vlm_backbone()
        return get_all_seq(packed_outputs)

    def _decode_action(self, packed_seq: PackedSequence,
                       last_hidden_state: torch.Tensor) -> List[torch.Tensor]:
        action = packed_seq.action
        if action is None:
            return []
        if action.mse_loss_indexes.numel() == 0:
            return [torch.zeros_like(tokens) for tokens in action.tokens]
        hidden_states = last_hidden_state[action.mse_loss_indexes]
        embodiment_ids = []
        for noisy_frame_indexes, embodiment_id in zip(
                action.noisy_frame_indexes, action.domain_id):
            embodiment_ids.append(
                embodiment_id.expand(len(noisy_frame_indexes)))
        per_token_embodiment_id = torch.cat(embodiment_ids, dim=0)
        preds_action = self.action_out_proj(hidden_states,
                                            per_token_embodiment_id)
        unpacked = self.unpack_action(preds_action, action.token_shapes,
                                      action.noisy_frame_indexes)
        if action.raw_action_dim is not None:
            for output, raw_dim in zip(unpacked, action.raw_action_dim):
                valid_dim = int(raw_dim.reshape(-1)[0].item())
                if valid_dim < output.shape[-1]:
                    output[..., valid_dim:] = 0
        return unpacked

    def _decode_vision(
        self,
        packed_seq: PackedSequence,
        last_hidden_state: torch.Tensor,
        original_latent_shapes: Optional[List[Tuple[int, int, int]]],
    ) -> List[torch.Tensor]:
        vision = packed_seq.vision
        if vision is None:
            return []
        if vision.mse_loss_indexes.numel() == 0:
            return [torch.zeros_like(tokens) for tokens in vision.tokens]
        preds_vision = self.vision_out_proj(
            last_hidden_state[vision.mse_loss_indexes])
        return self.unpatchify_and_unpack_latents(
            preds_vision,
            token_shapes_vision=vision.token_shapes,
            noisy_frame_indexes_vision=vision.noisy_frame_indexes,
            original_latent_shapes=original_latent_shapes,
        )

    @torch.no_grad()
    def decode_vision_latents(
        self,
        vision_latents: torch.Tensor | List[torch.Tensor],
    ) -> torch.Tensor | List[torch.Tensor]:
        is_list = not isinstance(vision_latents, torch.Tensor)
        latents = torch.cat(
            _as_list_of_1chw(vision_latents),
            dim=0) if is_list else vision_latents
        vae_latents = latents.to(
            device=self.vision_vae.device, dtype=self.vision_vae.dtype)
        decoded = self.vision_vae.decode(vae_latents)
        model_dtype = self._first_parameter(self.vision_in_proj).dtype
        decoded = decoded.to(device=self.device, dtype=model_dtype)
        if is_list:
            return [decoded[i:i + 1] for i in range(decoded.shape[0])]
        return decoded
