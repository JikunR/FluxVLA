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
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fluxvla.engines import HEADS
from fluxvla.wam_modes import (WAM_TRAINING_MODES, normalize_wam_mode_probs,
                               wam_mode_to_id)
from ..backbones.vlms.wan22_loader import build_action_dit, build_wan_video_dit
from ..third_party_models.fastwam.modules.mot import MoT
from ..third_party_models.fastwam.modules.schedulers.scheduler_continuous import \
    WanContinuousFlowMatchScheduler  # noqa: E501

__all__ = ['WAMHead']

_WAM_HEAD_COMPONENT_BUILDERS = {
    'WanVideoDiT': build_wan_video_dit,
    'ActionDiT': build_action_dit,
}


@HEADS.register_module()
class WAMHead(nn.Module):
    """WAM head with forward / IDM / policy / joint training branches.

    Owns the trainable components -- the ``video`` and ``action``
    experts wrapped by the :class:`MoT` mixed-attention module, the optional
    proprioception encoder, and the flow-matching schedulers -- together with
    training-loss and action-inference helpers.

    The video latents and ``context/context_mask`` tensors are produced
    upstream by ``WAMVLA`` from its regular ``vlm_backbone`` and separate
    ``video_latent_codec``, so this head consumes pre-encoded tensors.
    Training mode is usually sampled by the collator and passed in as
    ``training_mode``. FastWAM-aligned configs can instead set
    ``sample_mode_in_forward=True`` so rank 0 samples the step mode inside
    the head and broadcasts it across distributed ranks.
    """

    video_cond_noise_prob = 0.5

    def __init__(
        self,
        video_expert: Mapping[str, Any] | nn.Module,
        action_expert: Mapping[str, Any] | nn.Module,
        mot: Optional[nn.Module] = None,
        mot_checkpoint_mixed_attn: bool = True,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        temporal_downsample_factor: int = 4,
        skip_load: bool = False,
        video_scheduler: Optional[Dict[str, Any]] = None,
        action_scheduler: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_forward_video: float = 1.0,
        loss_lambda_idm_action: float = 1.0,
        loss_lambda_policy_action: float = 1.0,
        loss_lambda_joint_video: float = 1.0,
        loss_lambda_joint_action: float = 1.0,
        video_cond_noise_prob: float = 0.5,
        mode_probs: Optional[Dict[str, float]] = None,
        sample_mode_in_forward: bool = False,
        device: str = 'cpu',
        torch_dtype: torch.dtype = torch.float32,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        video_expert_config = self._extract_expert_config(
            video_expert, component_path='vla_head.video_expert')
        video_expert = self._build_component(
            video_expert,
            component_path='vla_head.video_expert',
            device=device,
            torch_dtype=torch_dtype,
            skip_load=skip_load,
        )
        action_expert = self._build_component(
            action_expert,
            component_path='vla_head.action_expert',
            device=device,
            torch_dtype=torch_dtype,
            skip_load=skip_load,
        )
        if mot is None:
            mot = MoT(
                mixtures={
                    'video': video_expert,
                    'action': action_expert,
                },
                mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
            )

        # Register only ``mot`` as a submodule; the experts live inside
        # ``mot.mixtures`` and are exposed via properties below. Registering
        # them again as ``self.video_expert`` / ``self.action_expert`` would
        # alias the same modules under two paths, which breaks FSDP's
        # recursive auto-wrap (a block would be wrapped twice). The
        # ``video_expert`` / ``action_expert`` args must be the very modules
        # held by ``mot`` so the property views stay consistent.
        if mot.mixtures['video'] is not video_expert \
                or mot.mixtures['action'] is not action_expert:
            raise ValueError(
                '`mot` must hold the same `video_expert` / `action_expert` '
                'instances passed to `WAMHead`.')
        self.mot = mot

        if text_dim is None:
            text_dim = self._infer_text_dim(video_expert, video_expert_config)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim,
                                             self.text_dim).to(
                                                 device=device,
                                                 dtype=torch_dtype)
        else:
            self.proprio_encoder = None

        self.temporal_downsample_factor = int(temporal_downsample_factor)

        video_scheduler = video_scheduler or {}
        action_scheduler = action_scheduler or {}
        loss = loss or {}
        video_train_shift = video_scheduler.get('train_shift',
                                                video_train_shift)
        video_infer_shift = video_scheduler.get('infer_shift',
                                                video_infer_shift)
        video_num_train_timesteps = video_scheduler.get(
            'num_train_timesteps', video_num_train_timesteps)
        action_train_shift = action_scheduler.get('train_shift',
                                                  action_train_shift)
        action_infer_shift = action_scheduler.get('infer_shift',
                                                  action_infer_shift)
        action_num_train_timesteps = action_scheduler.get(
            'num_train_timesteps', action_num_train_timesteps)
        loss_lambda_forward_video = loss.get(
            'lambda_forward_video',
            loss.get('lambda_video', loss_lambda_forward_video))
        loss_lambda_idm_action = loss.get(
            'lambda_idm_action',
            loss.get('lambda_action', loss_lambda_idm_action))
        loss_lambda_policy_action = loss.get(
            'lambda_policy_action',
            loss.get('lambda_action', loss_lambda_policy_action))
        loss_lambda_joint_video = loss.get(
            'lambda_joint_video',
            loss.get('lambda_video', loss_lambda_joint_video))
        loss_lambda_joint_action = loss.get(
            'lambda_joint_action',
            loss.get('lambda_action', loss_lambda_joint_action))

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_forward_video = float(loss_lambda_forward_video)
        self.loss_lambda_idm_action = float(loss_lambda_idm_action)
        self.loss_lambda_policy_action = float(loss_lambda_policy_action)
        self.loss_lambda_joint_video = float(loss_lambda_joint_video)
        self.loss_lambda_joint_action = float(loss_lambda_joint_action)
        self.video_cond_noise_prob = float(video_cond_noise_prob)
        self.mode_probs = normalize_wam_mode_probs(mode_probs)
        self.sample_mode_in_forward = bool(sample_mode_in_forward)

    @staticmethod
    def _extract_expert_config(component, component_path: str):
        if not isinstance(component, Mapping):
            return None
        config = component.get('config')
        if not isinstance(config, Mapping):
            raise ValueError(f'`{component_path}.config` must be a dict.')
        return dict(config)

    @staticmethod
    def _infer_text_dim(video_expert: nn.Module,
                        video_expert_config: Optional[Dict[str, Any]]) -> int:
        if (video_expert_config is not None
                and 'text_dim' in video_expert_config):
            return int(video_expert_config['text_dim'])
        text_dim = getattr(video_expert, 'text_dim', None)
        if text_dim is None:
            raise ValueError(
                '`vla_head.text_dim` is required when it cannot be inferred '
                'from `vla_head.video_expert.config.text_dim`.')
        return int(text_dim)

    @staticmethod
    def _build_component(component, *, component_path: str, device: str,
                         torch_dtype: torch.dtype, skip_load: bool):
        if isinstance(component, nn.Module):
            return component
        if not isinstance(component, Mapping):
            raise TypeError(
                f'`{component_path}` must be a dict or nn.Module, got '
                f'{type(component).__name__}.')
        cfg = dict(component)
        component_type = cfg.pop('type', None)
        if component_type is None:
            raise ValueError(f'`{component_path}.type` is required.')
        if isinstance(component_type, str):
            builder = _WAM_HEAD_COMPONENT_BUILDERS.get(component_type)
            if builder is None:
                raise KeyError(f'{component_type!r} is not a built-in WAM '
                               f'head component. For custom components, pass '
                               f'a callable in `{component_path}.type`.')
        elif callable(component_type):
            builder = component_type
        else:
            raise TypeError(
                f'`{component_path}.type` must be a string or callable, got '
                f'{type(component_type).__name__}.')
        cfg.setdefault('device', device)
        cfg.setdefault('torch_dtype', torch_dtype)
        if skip_load:
            cfg['skip_load_from_pretrain'] = True
        return builder(**cfg)

    # ``video_expert`` / ``action_expert`` are stored once inside
    # ``mot.mixtures`` (avoids submodule aliasing that breaks FSDP wrapping);
    # expose them as read-only views for the forward / inference logic.
    @property
    def video_expert(self) -> nn.Module:
        return self.mot.mixtures['video']

    @property
    def action_expert(self) -> nn.Module:
        return self.mot.mixtures['action']

    # ------------------------------------------------------------------
    # Helpers shared by WAM training and policy inference.
    # ------------------------------------------------------------------
    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ):
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError('`proprio` must be 2D [B, D], got shape '
                             f'{tuple(proprio.shape)}')
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(f'`proprio` last dim must be {self.proprio_dim}, '
                             f'got {proprio.shape[1]}')
        proprio_token = self.proprio_encoder(
            proprio.to(device=context.device,
                       dtype=context.dtype).unsqueeze(1)).to(
                           dtype=context.dtype)  # [B, 1, D]
        proprio_mask = torch.ones((context_mask.shape[0], 1),
                                  dtype=torch.bool,
                                  device=context_mask.device)
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len),
                           dtype=torch.bool,
                           device=device)

        mask[:video_seq_len, :video_seq_len] = \
            self.video_expert.build_video_to_video_mask(
                video_seq_len=video_seq_len,
                video_tokens_per_frame=video_tokens_per_frame,
                device=device,
            )
        mask[video_seq_len:, video_seq_len:] = True
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[video_seq_len:, :first_frame_tokens] = True
        return mask

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(
            pred_video.float(), target_video.float(),
            reduction='none').mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        temporal_factor = int(self.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError('`temporal_downsample_factor` must be positive, '
                             f'got {temporal_factor}.')
        if image_is_pad.shape[1] < 1:
            raise ValueError('`image_is_pad` must contain at least one frame.')
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                'Cannot align `image_is_pad` with video latent steps: '
                f'num_frames={image_is_pad.shape[1]}, '
                f'temporal_downsample_factor={temporal_factor}.')

        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1,
                                              temporal_factor).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad],
                                     dim=1)
        else:
            video_is_pad = latent_tail_is_pad

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError('Video-loss mask shape mismatch: '
                             f'mask steps={video_is_pad.shape[1]}, '
                             f'loss steps={video_loss_token.shape[1]}.')

        valid = (~video_is_pad).to(
            device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
        attention_mode: str = 'joint',
    ):
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask_kwargs = dict(
            video_seq_len=video_pre['tokens'].shape[1],
            action_seq_len=action_pre['tokens'].shape[1],
            video_tokens_per_frame=int(video_pre['meta']['tokens_per_frame']),
            device=video_pre['tokens'].device,
        )
        if attention_mode == 'joint':
            attention_mask = self._build_joint_attention_mask(
                **attention_mask_kwargs)
        elif attention_mode == 'forward':
            attention_mask = self._build_forward_attention_mask(
                **attention_mask_kwargs)
        else:
            raise ValueError(
                f'Unknown WAM attention mode: {attention_mode!r}.')
        tokens_out = self.mot(
            embeds_all={
                'video': video_pre['tokens'],
                'action': action_pre['tokens'],
            },
            attention_mask=attention_mask,
            freqs_all={
                'video': video_pre['freqs'],
                'action': action_pre['freqs'],
            },
            context_all={
                'video': {
                    'context': video_pre['context'],
                    'mask': video_pre['context_mask'],
                },
                'action': {
                    'context': action_pre['context'],
                    'mask': action_pre['context_mask'],
                },
            },
            t_mod_all={
                'video': video_pre['t_mod'],
                'action': action_pre['t_mod'],
            },
        )
        pred_video = self.video_expert.post_dit(tokens_out['video'], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out['action'],
                                                  action_pre)
        return pred_video, pred_action

    def _prepare_training_inputs(
        self,
        input_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        image_is_pad: Optional[torch.Tensor],
        proprio: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Move/validate training inputs and build the proprio-augmented
        context (no random draws), shared by the ``uncond`` and ``idm``
        forward passes so the noise-sampling order stays identical.
        """
        device = input_latents.device

        first_frame_latents = None
        fuse_flag = False
        if getattr(self.video_expert, 'fuse_vae_embedding_in_latents', False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                '`context/context_mask` must be [B,L,D]/[B,L], got '
                f'{tuple(context.shape)} and {tuple(context_mask.shape)}')
        context = context.to(
            device=device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(
            device=device, dtype=torch.bool, non_blocking=True)
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError(
                    '`proprio` is required when `proprio_dim` is enabled.')
            if proprio.ndim != 3:
                raise ValueError('`proprio` must be 3D [B, T, d], got shape '
                                 f'{tuple(proprio.shape)}')
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f'`proprio` last dim must be {self.proprio_dim}, '
                    f'got {proprio.shape[2]}')
            proprio = proprio[:, 0, :]  # [B, D]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=device, dtype=self.torch_dtype),
            )
        action = action.to(
            device=device, dtype=self.torch_dtype, non_blocking=True)
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(
                device=device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(
                device=device, dtype=torch.bool, non_blocking=True)

        return {
            'first_frame_latents': first_frame_latents,
            'fuse_flag': fuse_flag,
            'context': context,
            'context_mask': context_mask,
            'action': action,
            'action_is_pad': action_is_pad,
            'image_is_pad': image_is_pad,
        }

    # ------------------------------------------------------------------
    # Action inference denoising loop.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_expert_with_cache(
            expert_name='action',
            tokens=action_pre['tokens'],
            freqs=action_pre['freqs'],
            t_mod=action_pre['t_mod'],
            context_payload={
                'context': action_pre['context'],
                'mask': action_pre['context_mask'],
            },
            kv_cache_by_expert={'video': video_kv_cache},
            attention_mask=attention_mask,
            attention_order=('video', 'action'),
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()
    def predict_action(
        self,
        first_frame_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action_horizon: int,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = 'cpu',
        **kwargs,
    ) -> torch.Tensor:
        self.eval()
        if str(getattr(self.video_expert, 'video_attention_mask_mode', '')) \
                != 'first_frame_causal':
            raise ValueError(
                '`predict_action` requires '
                "`video_attention_mask_mode='first_frame_causal'`.")

        device = first_frame_latents.device
        generator = (None if seed is None else torch.Generator(
            device=rand_device).manual_seed(seed))
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(
            device=device, dtype=self.torch_dtype)

        fuse_flag = bool(
            getattr(self.video_expert, 'fuse_vae_embedding_in_latents', False))

        timestep_video = torch.zeros(
            (first_frame_latents.shape[0], ),
            dtype=first_frame_latents.dtype,
            device=device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre['tokens'].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre['meta']['tokens_per_frame']),
            device=video_pre['tokens'].device,
        )
        video_kv_cache = self.mot.prefill_expert_cache(
            expert_name='video',
            tokens=video_pre['tokens'],
            freqs=video_pre['freqs'],
            t_mod=video_pre['t_mod'],
            context_payload={
                'context': video_pre['context'],
                'mask': video_pre['context_mask'],
            },
            attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        infer_timesteps_action, infer_deltas_action = \
            self.infer_action_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=device,
                dtype=latents_action.dtype,
                shift_override=sigma_shift,
            )
        schedule = zip(infer_timesteps_action, infer_deltas_action)
        for step_t_action, step_delta_action in schedule:
            timestep_action = step_t_action.unsqueeze(0).to(
                dtype=latents_action.dtype, device=device)
            pred_action = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
            )
            latents_action = self.infer_action_scheduler.step(
                pred_action, step_delta_action, latents_action)

        return latents_action

    @torch.no_grad()
    def predict_video_action(
        self,
        first_frame_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action_horizon: int,
        video_latent_shape,
        action: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = 'cpu',
        **kwargs,
    ):
        self.eval()
        device = first_frame_latents.device
        z_dim, latent_t, latent_h, latent_w = video_latent_shape

        video_generator = (None if seed is None else torch.Generator(
            device=rand_device).manual_seed(seed))
        action_generator = (None if seed is None else torch.Generator(
            device=rand_device).manual_seed(seed))
        latents_video = torch.randn(
            (1, z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(
            device=device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(
            device=device, dtype=self.torch_dtype)
        latents_video[:, :, 0:1] = first_frame_latents.clone()

        fixed_action = None
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if (action.ndim != 3 or action.shape[0] != 1
                    or action.shape[1] != action_horizon):
                raise ValueError(
                    '`action` must have shape [T, D] or [1, T, D] '
                    f'with action_horizon={action_horizon}, got '
                    f'{tuple(action.shape)}')
            fixed_action = action.to(device=device, dtype=self.torch_dtype)

        fuse_flag = bool(
            getattr(self.video_expert, 'fuse_vae_embedding_in_latents', False))
        infer_timesteps_video, infer_deltas_video = \
            self.infer_video_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=device,
                dtype=latents_video.dtype,
                shift_override=sigma_shift,
            )
        infer_timesteps_action, infer_deltas_action = \
            self.infer_action_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=device,
                dtype=latents_action.dtype,
                shift_override=sigma_shift,
            )
        for step_t_video, step_delta_video, step_t_action, step_delta_action \
                in zip(infer_timesteps_video, infer_deltas_video,
                       infer_timesteps_action, infer_deltas_action):
            timestep_video = step_t_video.unsqueeze(0).to(
                dtype=latents_video.dtype, device=device)
            timestep_action = step_t_action.unsqueeze(0).to(
                dtype=latents_action.dtype, device=device)
            if fixed_action is not None:
                latents_action = fixed_action
                timestep_action = torch.zeros_like(timestep_action)
            pred_video, pred_action = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=fixed_action,
                attention_mode='forward'
                if fixed_action is not None else 'joint',
            )
            latents_video = self.infer_video_scheduler.step(
                pred_video, step_delta_video, latents_video)
            if fixed_action is None:
                latents_action = self.infer_action_scheduler.step(
                    pred_action, step_delta_action, latents_action)
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        return latents_video, latents_action[0].detach().to(
            device='cpu', dtype=torch.float32)

    @classmethod
    def _prepare_training_mode_ids(
        cls,
        training_mode,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if training_mode is None:
            raise ValueError(
                '`training_mode` must be provided by the WAM data pipeline.')

        if isinstance(training_mode, str):
            ids = torch.full((batch_size, ),
                             wam_mode_to_id(training_mode),
                             dtype=torch.long,
                             device=device)
        elif isinstance(training_mode, (list, tuple)):
            if len(training_mode) != batch_size:
                raise ValueError('`training_mode` length must match batch '
                                 f'size {batch_size}, got '
                                 f'{len(training_mode)}.')
            ids = [
                wam_mode_to_id(mode) if isinstance(mode, str) else int(mode)
                for mode in training_mode
            ]
            ids = torch.tensor(ids, dtype=torch.long, device=device)
        else:
            ids = torch.as_tensor(
                training_mode, dtype=torch.long, device=device)
            if ids.ndim == 0:
                ids = ids.expand(batch_size)
            else:
                ids = ids.reshape(-1)
            if ids.numel() != batch_size:
                raise ValueError('`training_mode` size must match batch '
                                 f'size {batch_size}, got {ids.numel()}.')

        if ids.numel() > 0:
            min_id = int(ids.min().item())
            max_id = int(ids.max().item())
            if min_id < 0 or max_id >= len(WAM_TRAINING_MODES):
                raise ValueError(
                    '`training_mode` contains invalid mode id(s): '
                    f'min={min_id}, max={max_id}.')
        return ids

    @torch.no_grad()
    def _build_forward_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len),
                           dtype=torch.bool,
                           device=device)

        mask[:video_seq_len, :video_seq_len] = \
            self.video_expert.build_video_to_video_mask(
                video_seq_len=video_seq_len,
                video_tokens_per_frame=video_tokens_per_frame,
                device=device,
            )
        mask[video_seq_len:, video_seq_len:] = True
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[first_frame_tokens:video_seq_len, video_seq_len:] = True
        return mask

    @torch.no_grad()
    def _build_idm_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len),
                           dtype=torch.bool,
                           device=device)

        mask[:video_seq_len, :video_seq_len] = \
            self.video_expert.build_video_to_video_mask(
                video_seq_len=video_seq_len,
                video_tokens_per_frame=video_tokens_per_frame,
                device=device,
            )
        mask[video_seq_len:, video_seq_len:] = True
        mask[video_seq_len:, :video_seq_len] = True
        return mask

    @torch.no_grad()
    def _build_policy_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len),
                           dtype=torch.bool,
                           device=device)

        mask[:video_seq_len, :video_seq_len] = \
            self.video_expert.build_video_to_video_mask(
                video_seq_len=video_seq_len,
                video_tokens_per_frame=video_tokens_per_frame,
                device=device,
            )
        mask[video_seq_len:, video_seq_len:] = True
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[video_seq_len:, :first_frame_tokens] = True
        return mask

    @torch.no_grad()
    def _build_joint_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len),
                           dtype=torch.bool,
                           device=device)

        mask[:video_seq_len, :video_seq_len] = \
            self.video_expert.build_video_to_video_mask(
                video_seq_len=video_seq_len,
                video_tokens_per_frame=video_tokens_per_frame,
                device=device,
            )
        mask[video_seq_len:, video_seq_len:] = True
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[first_frame_tokens:video_seq_len, video_seq_len:] = True
        mask[video_seq_len:, :video_seq_len] = True
        return mask

    @torch.no_grad()
    def _build_training_attention_mask(
        self,
        mode_ids: torch.Tensor,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        forward_mask = self._build_forward_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=action_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        idm_mask = self._build_idm_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=action_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        policy_mask = self._build_policy_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=action_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        joint_mask = self._build_joint_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=action_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        masks = torch.stack([forward_mask, idm_mask, policy_mask, joint_mask],
                            dim=0)
        mode_ids = mode_ids.to(device=device, dtype=torch.long)
        has_modes = mode_ids.numel() > 0
        if has_modes and bool((mode_ids == mode_ids[0]).all().item()):
            return masks[int(mode_ids[0].item())]
        return masks.index_select(0, mode_ids).unsqueeze(1)

    def _compute_action_loss_per_sample(
        self,
        pred_action: torch.Tensor,
        target_action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        action_loss_token = F.mse_loss(
            pred_action.float(),
            target_action.float(),
            reduction='none',
        ).mean(dim=2)
        if action_is_pad is not None:
            valid = (~action_is_pad).to(
                device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            return (action_loss_token * valid).sum(dim=1) / valid_sum
        return action_loss_token.mean(dim=1)

    def _sample_training_mode_ids(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        probs = torch.tensor(
            [self.mode_probs[mode] for mode in WAM_TRAINING_MODES],
            device=device,
            dtype=torch.float32,
        )
        idx_tensor = torch.empty((), device=device, dtype=torch.long)
        if torch.distributed.is_available() \
                and torch.distributed.is_initialized():
            backend = str(torch.distributed.get_backend()).lower()
            idx_device = torch.device('cpu') if backend == 'gloo' else device
            idx_tensor = torch.empty((), device=idx_device, dtype=torch.long)
            if torch.distributed.get_rank() == 0:
                sampled = torch.multinomial(
                    probs, num_samples=1).to(
                        device=idx_device, dtype=torch.long)
                idx_tensor.copy_(sampled[0])
            torch.distributed.broadcast(idx_tensor, src=0)
            idx_tensor = idx_tensor.to(device=device)
        else:
            idx_tensor.copy_(torch.multinomial(probs, num_samples=1)[0])
        return idx_tensor.expand(batch_size)

    def forward(
        self,
        input_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor] = None,
        image_is_pad: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        training_mode: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        prep = self._prepare_training_inputs(
            input_latents=input_latents,
            context=context,
            context_mask=context_mask,
            action=action,
            action_is_pad=action_is_pad,
            image_is_pad=image_is_pad,
            proprio=proprio,
        )
        inputs = dict(prep)

        if self.sample_mode_in_forward:
            mode_ids = self._sample_training_mode_ids(
                batch_size=int(input_latents.shape[0]),
                device=input_latents.device,
            )
        else:
            mode_ids = self._prepare_training_mode_ids(
                training_mode=training_mode,
                batch_size=int(input_latents.shape[0]),
                device=input_latents.device,
            )

        batch_size_int = int(input_latents.shape[0])
        batch_size = float(batch_size_int)
        device = input_latents.device

        is_forward = mode_ids == wam_mode_to_id('forward')
        is_idm = mode_ids == wam_mode_to_id('idm')
        is_policy = mode_ids == wam_mode_to_id('policy')
        is_joint = mode_ids == wam_mode_to_id('joint')

        video_noise = torch.randn_like(input_latents)
        sampled_timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size_int,
            device=device,
            dtype=input_latents.dtype,
        )
        noisy_video = self.train_video_scheduler.add_noise(
            input_latents, video_noise, sampled_timestep_video)
        target_video = self.train_video_scheduler.training_target(
            input_latents, video_noise, sampled_timestep_video)

        zero_timestep_video = torch.zeros((batch_size_int, ),
                                          dtype=input_latents.dtype,
                                          device=device)
        idm_cond_mask = (
            torch.rand((batch_size_int, ), device=device) < float(
                self.video_cond_noise_prob))
        idm_timestep_video_sampled = \
            self.train_video_scheduler.sample_training_t(
                batch_size=batch_size_int,
                device=device,
                dtype=input_latents.dtype,
            )
        idm_timestep_video = torch.where(idm_cond_mask,
                                         idm_timestep_video_sampled,
                                         zero_timestep_video)
        idm_video_noise = torch.randn_like(input_latents)
        idm_noisy_video = self.train_video_scheduler.add_noise(
            input_latents, idm_video_noise, idm_timestep_video_sampled)
        idm_video_selector = idm_cond_mask.view(batch_size_int, 1, 1, 1, 1)
        idm_video = torch.where(idm_video_selector, idm_noisy_video,
                                input_latents)

        video_selector = (is_forward | is_joint).view(batch_size_int, 1, 1, 1,
                                                      1)
        latents_video = torch.where(video_selector, noisy_video, input_latents)
        latents_video = torch.where(
            is_idm.view(batch_size_int, 1, 1, 1, 1), idm_video, latents_video)
        timestep_video = torch.where(is_forward | is_joint,
                                     sampled_timestep_video,
                                     zero_timestep_video)
        timestep_video = torch.where(is_idm, idm_timestep_video,
                                     timestep_video)

        if inputs['first_frame_latents'] is not None:
            latents_video = torch.cat(
                [inputs['first_frame_latents'], latents_video[:, :, 1:]],
                dim=2,
            )

        action = inputs['action']
        action_noise = torch.randn_like(action)
        sampled_timestep_action = \
            self.train_action_scheduler.sample_training_t(
                batch_size=batch_size_int,
                device=device,
                dtype=action.dtype,
            )
        noisy_action = self.train_action_scheduler.add_noise(
            action, action_noise, sampled_timestep_action)
        target_action = self.train_action_scheduler.training_target(
            action, action_noise, sampled_timestep_action)
        zero_timestep_action = torch.zeros((batch_size_int, ),
                                           dtype=action.dtype,
                                           device=device)
        latents_action = torch.where(
            is_forward.view(batch_size_int, 1, 1), action, noisy_action)
        timestep_action = torch.where(is_forward, zero_timestep_action,
                                      sampled_timestep_action)

        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=inputs['context'],
            context_mask=inputs['context_mask'],
            action=(action if getattr(self.video_expert, 'action_conditioned',
                                      False) else None),
            fuse_vae_embedding_in_latents=inputs['fuse_flag'],
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=inputs['context'],
            context_mask=inputs['context_mask'],
        )

        attention_mask = self._build_training_attention_mask(
            mode_ids=mode_ids,
            video_seq_len=video_pre['tokens'].shape[1],
            action_seq_len=action_pre['tokens'].shape[1],
            video_tokens_per_frame=int(video_pre['meta']['tokens_per_frame']),
            device=video_pre['tokens'].device,
        )
        tokens_out = self.mot(
            embeds_all={
                'video': video_pre['tokens'],
                'action': action_pre['tokens'],
            },
            attention_mask=attention_mask,
            freqs_all={
                'video': video_pre['freqs'],
                'action': action_pre['freqs'],
            },
            context_all={
                'video': {
                    'context': video_pre['context'],
                    'mask': video_pre['context_mask'],
                },
                'action': {
                    'context': action_pre['context'],
                    'mask': action_pre['context_mask'],
                },
            },
            t_mod_all={
                'video': video_pre['t_mod'],
                'action': action_pre['t_mod'],
            },
        )
        pred_video = self.video_expert.post_dit(tokens_out['video'], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out['action'],
                                                  action_pre)

        include_initial_video_step = inputs['first_frame_latents'] is None
        if inputs['first_frame_latents'] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        video_loss_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=inputs['image_is_pad'],
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(
            sampled_timestep_video).to(
                video_loss_per_sample.device,
                dtype=video_loss_per_sample.dtype,
            )
        weighted_video_loss = video_loss_per_sample * video_weight

        action_loss_per_sample = self._compute_action_loss_per_sample(
            pred_action=pred_action,
            target_action=target_action,
            action_is_pad=inputs['action_is_pad'],
        )
        action_weight = self.train_action_scheduler.training_weight(
            sampled_timestep_action).to(
                action_loss_per_sample.device,
                dtype=action_loss_per_sample.dtype,
            )
        weighted_action_loss = action_loss_per_sample * action_weight

        video_dtype = weighted_video_loss.dtype
        action_dtype = weighted_action_loss.dtype
        forward_selector = is_forward.to(device=device, dtype=video_dtype)
        idm_selector = is_idm.to(device=device, dtype=action_dtype)
        policy_selector = is_policy.to(device=device, dtype=action_dtype)
        joint_video_selector = is_joint.to(device=device, dtype=video_dtype)
        joint_action_selector = is_joint.to(device=device, dtype=action_dtype)

        loss_forward_video = (
            self.loss_lambda_forward_video *
            (weighted_video_loss * forward_selector).sum() / batch_size)
        loss_idm_action = (
            self.loss_lambda_idm_action *
            (weighted_action_loss * idm_selector).sum() / batch_size)
        loss_policy_action = (
            self.loss_lambda_policy_action *
            (weighted_action_loss * policy_selector).sum() / batch_size)
        loss_joint_video = (
            self.loss_lambda_joint_video *
            (weighted_video_loss * joint_video_selector).sum() / batch_size)
        loss_joint_action = (
            self.loss_lambda_joint_action *
            (weighted_action_loss * joint_action_selector).sum() / batch_size)
        loss_total = (
            loss_forward_video + loss_idm_action + loss_policy_action +
            loss_joint_video + loss_joint_action)

        return {
            'loss': loss_total,
            'loss_forward_video': loss_forward_video.detach(),
            'loss_idm_action': loss_idm_action.detach(),
            'loss_policy_action': loss_policy_action.detach(),
            'loss_joint_video': loss_joint_video.detach(),
            'loss_joint_action': loss_joint_action.detach(),
        }
