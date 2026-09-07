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

from typing import Any, Callable, Dict, Optional

import torch
from PIL import Image

from fluxvla.engines import VLAS, initialize_overwatch
from fluxvla.engines.utils.name_map import str_to_dtype
from ..backbones.vlms.wan22_loader import build_wan_video_vae38
from ..heads.wam_head import WAMHead
from .base_vla import BaseVLA

overwatch = initialize_overwatch(__name__)

__all__ = ['WAMVLA']

_WAM_COMPONENT_BUILDERS = {
    'Wan22VAE': build_wan_video_vae38,
}


@VLAS.register_module()
class WAMVLA(BaseVLA):
    """Minimal WAM built in FluxVLA's backbone/head style.

    Wraps WAM world/action modules as a FluxVLA VLA composed of:

    * ``vlm_backbone`` -- an optional frozen language/VLM context backbone,
      following the normal FluxVLA meaning of that field. Training can omit it
      when the dataloader supplies cached ``context/context_mask`` tensors.
    * ``video_latent_codec`` -- the frozen video VAE used to encode
      observations into WAM latents and decode joint video predictions.
    * :class:`~fluxvla.models.heads.wam_head.WAMHead`
      (``vla_head``) -- video/action experts, MoT mixed attention,
      flow-matching schedulers and sampled forward / IDM / policy objectives.

    The VAE is kept outside ``vlm_backbone`` so switching Wan text encoder,
    QwenVL, or future LLM/VLM context sources stays a pure backbone choice.
    When online context encoding is needed, WAM calls the backbone through the
    normal ``forward(images, lang_tokens, img_masks, lang_masks, ...)``
    interface and adapts the returned hidden states into the
    ``context/context_mask`` pair consumed by the head.

    Training follows the shared FluxVLA batch contract:
    ``images`` is ``[B, 3, T, H, W]`` after ``PrepareVideo``,
    ``context/context_mask`` hold cached text features unless online
    ``lang_tokens/lang_masks`` are provided, ``states`` holds proprioception,
    ``actions`` holds the target action window, and ``action_masks`` /
    ``frame_masks`` use ``True`` for valid entries.
    """

    def __init__(
        self,
        vlm_backbone: Optional[Dict] = None,
        video_latent_codec: Optional[Dict] = None,
        vla_head: Optional[Dict] = None,
        proprio_dim: Optional[int] = None,
        action_horizon: Optional[int] = None,
        frame_window_size: Optional[int] = None,
        num_views: Optional[int] = None,
        mot_checkpoint_mixed_attn: bool = True,
        skip_load: bool = False,
        pretrained_name_or_path: Optional[str] = None,
        name_mapping: Optional[Dict] = None,
        strict_mapping: bool = False,
        freeze_vlm_backbone: bool = True,
        device: str = 'cpu',
        torch_dtype: torch.dtype = torch.float32,
        *args,
        **kwargs,
    ) -> None:
        if video_latent_codec is None:
            raise ValueError('`video_latent_codec` is required for WAMVLA.')
        if vla_head is None:
            raise ValueError('`vla_head` is required for WAMVLA.')
        self._build_device = str(device)
        if isinstance(torch_dtype, str):
            torch_dtype = str_to_dtype(torch_dtype)
        self.torch_dtype = torch_dtype
        proprio_dim_value = None if proprio_dim is None else int(proprio_dim)
        action_horizon_value = (None if action_horizon is None else
                                int(action_horizon))
        num_views_value = None if num_views is None else int(num_views)
        frame_window_size_value = (None if frame_window_size is None else
                                   int(frame_window_size))

        backbone_cfg = None if vlm_backbone is None else dict(vlm_backbone)
        if skip_load and backbone_cfg is not None:
            backbone_cfg.setdefault('skip_load', True)
        video_latent_codec_module = self._build_wam_component(
            self._pop_component_cfg(
                {'video_latent_codec': video_latent_codec},
                key='video_latent_codec',
                owner_name='model',
            ),
            component_path='video_latent_codec',
            device=self._build_device,
            torch_dtype=self.torch_dtype,
            skip_load=skip_load,
        )
        head_cfg = dict(vla_head)
        head_cfg.setdefault('proprio_dim', proprio_dim_value)
        head_cfg.setdefault(
            'temporal_downsample_factor',
            int(video_latent_codec_module.temporal_downsample_factor),
        )
        head_cfg.setdefault('skip_load', skip_load)
        head_cfg.setdefault('device', self._build_device)
        head_cfg.setdefault('torch_dtype', self.torch_dtype)
        head_cfg.setdefault('mot_checkpoint_mixed_attn',
                            bool(mot_checkpoint_mixed_attn))
        super().__init__(
            vlm_backbone=backbone_cfg,
            vla_head=head_cfg,
            freeze_vlm_backbone=freeze_vlm_backbone,
            pretrained_name_or_path=None,
            name_mapping=name_mapping,
            strict_mapping=strict_mapping,
        )
        self.proprio_dim = proprio_dim_value
        self.action_horizon = action_horizon_value
        self.num_views = num_views_value
        self.frame_window_size = frame_window_size_value
        self.video_latent_codec = video_latent_codec_module
        self.video_latent_codec.requires_grad_(False)

        self.all_module_keys = ['video_latent_codec', 'vla_head']
        if self.vlm_backbone is not None:
            self.all_module_keys.insert(0, 'vlm_backbone')

        if pretrained_name_or_path is not None:
            self.pretrained_name_or_path = pretrained_name_or_path
            self.from_pretrained()

    @property
    def input_encoder(self):
        return self.vlm_backbone

    def freeze_backbones(self) -> None:
        super().freeze_backbones()
        self.video_latent_codec.requires_grad_(False)
        self.video_latent_codec.eval()

    @property
    def video_latent_codec_device(self) -> torch.device:
        for tensor in self.video_latent_codec.parameters():
            return tensor.device
        for tensor in self.video_latent_codec.buffers():
            return tensor.device
        return torch.device(self._build_device)

    def _prepare_vlm_images(self, images: Optional[torch.Tensor]):
        if images is None:
            return None, None
        if images.ndim == 5:
            if images.shape[1] != 3:
                raise ValueError(
                    '`images` must be [B, 3, T, H, W] for WAM, got '
                    f'{tuple(images.shape)}')
            context_images = images[:, :, 0]
        elif images.ndim == 4:
            context_images = images
        elif images.ndim == 3:
            context_images = images.unsqueeze(0)
        else:
            raise ValueError(
                '`images` for vlm_backbone must be 3D, 4D or 5D, got '
                f'{tuple(images.shape)}')
        context_images = context_images.unsqueeze(1)
        context_img_masks = torch.ones(
            (context_images.shape[0], context_images.shape[1]),
            dtype=torch.bool,
            device=context_images.device,
        )
        return context_images, context_img_masks

    @staticmethod
    def _normalize_vlm_context_output(outputs, lang_masks=None):
        if isinstance(outputs, dict):
            context = outputs.get('last_hidden_state')
            if context is None and 'hidden_states' in outputs:
                context = outputs['hidden_states'][-1]
            context_mask = outputs.get('attention_mask')
        else:
            if not isinstance(outputs, (list, tuple)) or len(outputs) < 2:
                raise TypeError('`vlm_backbone.forward` must return '
                                '`(context, context_mask, ...)` or a dict.')
            context, context_mask = outputs[0], outputs[1]
        if context is None:
            raise ValueError('`vlm_backbone.forward` did not return context.')
        if context.ndim != 3:
            raise ValueError('`context` must be 3D [B, L, D], got shape '
                             f'{tuple(context.shape)}')

        if (isinstance(context_mask, torch.Tensor) and context_mask.ndim == 2
                and tuple(context_mask.shape) == tuple(context.shape[:2])):
            return context, context_mask.to(
                device=context.device, dtype=torch.bool)

        mask = torch.ones(
            context.shape[:2], dtype=torch.bool, device=context.device)
        if lang_masks is None:
            return context, mask

        lang_masks = lang_masks.to(device=context.device, dtype=torch.bool)
        if lang_masks.ndim == 1:
            lang_masks = lang_masks.unsqueeze(0)
        if (lang_masks.shape[0] != context.shape[0]
                or lang_masks.shape[1] > context.shape[1]):
            return context, mask
        text_start = context.shape[1] - lang_masks.shape[1]
        mask[:, text_start:] = lang_masks
        context = context.clone()
        context[:, text_start:] = context[:, text_start:].masked_fill(
            ~lang_masks.unsqueeze(-1), 0)
        return context, mask

    def _encode_vlm_context(
        self,
        images: Optional[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor] = None,
    ):
        if self.vlm_backbone is None:
            raise ValueError(
                'WAMVLA has no `vlm_backbone`; pass precomputed '
                '`context/context_mask` or configure `inference_model` with a '
                'context backbone.')
        if hasattr(self.vlm_backbone, 'set_frozen_modules_to_eval_mode'):
            self.vlm_backbone.set_frozen_modules_to_eval_mode()
        context_images, context_img_masks = self._prepare_vlm_images(images)
        outputs = self.vlm_backbone(
            images=context_images,
            lang_tokens=lang_tokens,
            img_masks=context_img_masks,
            lang_masks=lang_masks,
            image_grid_thw=image_grid_thw,
        )
        return self._normalize_vlm_context_output(outputs, lang_masks)

    @torch.no_grad()
    def _encode_video_latents(
            self,
            video_tensor,
            tiled: bool = False,
            tile_size=(30, 52),
            tile_stride=(15, 26),
    ):
        self.video_latent_codec.eval()
        return self.video_latent_codec.encode(
            video_tensor,
            device=self.video_latent_codec_device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )

    @torch.no_grad()
    def _encode_input_image_latents(
            self,
            input_image: torch.Tensor,
            tiled: bool = False,
            tile_size=(30, 52),
            tile_stride=(15, 26),
    ):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if (input_image.ndim != 4 or input_image.shape[0] != 1
                or input_image.shape[1] != 3):
            raise ValueError(
                '`input_image` must have shape [1,3,H,W] or [3,H,W], got '
                f'{tuple(input_image.shape)}')
        self.video_latent_codec.eval()
        image = input_image.to(
            device=self.video_latent_codec_device)[0].unsqueeze(1)
        z = self.video_latent_codec.encode(
            [image],
            device=self.video_latent_codec_device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    def _build_video_latent_shape(self, input_image: torch.Tensor,
                                  num_frames: int):
        height, width = int(input_image.shape[-2]), int(input_image.shape[-1])
        codec = self.video_latent_codec
        z_dim = int(codec.model.z_dim)
        temporal_factor = int(codec.temporal_downsample_factor)
        upsampling_factor = int(codec.upsampling_factor)
        latent_t = (int(num_frames) - 1) // temporal_factor + 1
        latent_h = height // upsampling_factor
        latent_w = width // upsampling_factor
        return z_dim, latent_t, latent_h, latent_w

    def _decode_latents(
            self,
            latents,
            tiled: bool = False,
            tile_size=(30, 52),
            tile_stride=(15, 26),
    ):
        video_tensor = self.video_latent_codec.decode(
            latents,
            device=self.video_latent_codec_device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    @staticmethod
    def _pop_component_cfg(owner_cfg, key, owner_name, required=True):
        cfg = owner_cfg.pop(key, None)
        if cfg is None:
            if required:
                raise ValueError(
                    f'`{owner_name}.{key}` is required for WAMVLA.')
            return None
        if not isinstance(cfg, dict):
            raise TypeError(f'`{owner_name}.{key}` must be a dict.')
        if 'type' not in cfg:
            raise ValueError(f'`{owner_name}.{key}.type` is required.')
        return dict(cfg)

    @staticmethod
    def _build_wam_component(component_cfg: Dict[str, Any], *,
                             component_path: str, device: str,
                             torch_dtype: torch.dtype, skip_load: bool):
        cfg = dict(component_cfg)
        component_type = cfg.pop('type')
        if isinstance(component_type, str):
            builder = _WAM_COMPONENT_BUILDERS.get(component_type)
            if builder is None:
                raise KeyError(f'{component_type!r} is not a built-in WAM '
                               f'component. For custom components, '
                               f'pass a callable in `{component_path}.type`.')
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

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        lang_tokens: Optional[torch.Tensor] = None,
        lang_masks: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        action_masks: Optional[torch.Tensor] = None,
        state_chunks: Optional[torch.Tensor] = None,
        state_chunk_masks: Optional[torch.Tensor] = None,
        frame_masks: Optional[torch.Tensor] = None,
        training_mode: Optional[torch.Tensor] = None,
        embodiment_ids: Optional[torch.Tensor] = None,
        img_masks: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if (context is None and context_mask is None
                and lang_tokens is not None):
            if lang_masks is None:
                raise ValueError(
                    '`lang_masks` must be provided with `lang_tokens`.')
            context, context_mask = self._encode_vlm_context(
                images=images,
                lang_tokens=lang_tokens,
                lang_masks=lang_masks,
                image_grid_thw=kwargs.get('image_grid_thw'),
            )
        if images is None or context is None or context_mask is None \
                or actions is None:
            raise ValueError(
                'WAMVLA.forward requires `images`, `actions` and '
                'either `context/context_mask` or `lang_tokens/lang_masks`.')
        if images.ndim != 5:
            raise ValueError('`images` must be 5D [B, 3, T, H, W], got shape '
                             f'{tuple(images.shape)}')
        if images.shape[1] != 3:
            raise ValueError('`images` channel dimension must be 3, got shape '
                             f'{tuple(images.shape)}')
        _, _, num_frames, height, width = images.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError('Video spatial dims must be multiples of 16, got '
                             f'H={height}, W={width}')
        if num_frames % 4 != 1:
            raise ValueError(
                f'Video T must satisfy T % 4 == 1, got {num_frames}')
        if num_frames <= 1:
            raise ValueError(
                f'Video T must be > 1 for action-conditioned training, '
                f'got {num_frames}')

        images = images.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True)
        input_latents = self._encode_video_latents(images, tiled=False)
        proprio = states
        if proprio is not None and proprio.ndim == 2:
            proprio = proprio.unsqueeze(1)
        action_is_pad = None
        if action_masks is not None:
            action_is_pad = ~action_masks.to(dtype=torch.bool)
        state_chunk_is_pad = None
        if state_chunk_masks is not None:
            state_chunk_is_pad = ~state_chunk_masks.to(dtype=torch.bool)
        image_is_pad = None
        if frame_masks is not None:
            image_is_pad = ~frame_masks.to(dtype=torch.bool)

        return self.vla_head(
            input_latents=input_latents,
            context=context,
            context_mask=context_mask,
            action=actions,
            action_is_pad=action_is_pad,
            image_is_pad=image_is_pad,
            proprio=proprio,
            training_mode=training_mode,
            state_chunks=state_chunks,
            state_chunk_is_pad=state_chunk_is_pad,
            embodiment_ids=embodiment_ids,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_action(
        self,
        input_image: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        lang_tokens: Optional[torch.Tensor] = None,
        lang_masks: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
        embodiment_ids: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = 'cpu',
        tiled: bool = False,
        return_state_chunks: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if 'prompt' in kwargs:
            raise ValueError(
                'WAM expects tokenized prompts from the data pipeline. Pass '
                '`lang_tokens/lang_masks` or precomputed '
                '`context/context_mask`.')
        # Adapt the shared ``LiberoParquetEvalDataset`` batch
        # (images / lang_tokens / lang_masks / states) to WAM inputs.
        # Explicit ``input_image`` / ``context`` / ``proprio`` take priority so
        # callers can still pass precomputed context when desired.
        if input_image is None and images is not None:
            if images.ndim != 5:
                raise ValueError('`images` must be 5D [B, C, T, H, W], got '
                                 f'{tuple(images.shape)}')
            input_image = images[:, :, 0]
        if proprio is None and states is not None:
            proprio = states
        if (context is None and context_mask is None
                and lang_tokens is not None):
            if lang_masks is None:
                raise ValueError(
                    '`lang_masks` must be provided with `lang_tokens`.')
            context, context_mask = self._encode_vlm_context(
                images=input_image,
                lang_tokens=lang_tokens,
                lang_masks=lang_masks,
                image_grid_thw=kwargs.get('image_grid_thw'),
            )
        if action_horizon is None:
            action_horizon = self.action_horizon
        if action_horizon is None:
            raise ValueError(
                '`action_horizon` must be provided or configured on the '
                'model via `action_horizon=`.')
        if input_image is None:
            raise ValueError(
                'predict_action requires `input_image` or `images`.')
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        input_image = input_image.to(
            device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents(
            input_image, tiled=tiled)

        use_context = context is not None or context_mask is not None
        if not use_context:
            raise ValueError('Either `lang_tokens/lang_masks` or both '
                             '`context/context_mask` must be provided.')
        if context is None or context_mask is None:
            raise ValueError(
                '`context` and `context_mask` must be provided together.')
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)
        context = context.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(
            device=self.device, dtype=torch.bool, non_blocking=True)

        if proprio is not None and self.vla_head.proprio_encoder is not None:
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)
            context, context_mask = self.vla_head._append_proprio_to_context(
                context=context, context_mask=context_mask, proprio=proprio)

        # Joint / idm heads denoise a full imagined video, so they need the
        # video latent shape. ``frame_window_size`` must be configured for
        # those variants; the uncond head ignores ``video_latent_shape``.
        video_latent_shape = None
        if self.frame_window_size is not None:
            video_latent_shape = self._build_video_latent_shape(
                input_image=input_image,
                num_frames=self.frame_window_size,
            )

        predict_kwargs = dict(
            first_frame_latents=first_frame_latents,
            context=context,
            context_mask=context_mask,
            action_horizon=action_horizon,
            video_latent_shape=video_latent_shape,
            proprio=proprio,
            embodiment_ids=embodiment_ids,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
        )
        if return_state_chunks:
            predict_kwargs['return_state_chunks'] = True
        return self.vla_head.predict_action(**predict_kwargs)

    def _prepare_inference_context(
        self,
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
        images: Optional[torch.Tensor],
        lang_tokens: Optional[torch.Tensor],
        lang_masks: Optional[torch.Tensor],
        proprio: Optional[torch.Tensor],
    ):
        if (context is None and context_mask is None
                and lang_tokens is not None):
            if lang_masks is None:
                raise ValueError(
                    '`lang_masks` must be provided with `lang_tokens`.')
            context, context_mask = self._encode_vlm_context(
                images=images,
                lang_tokens=lang_tokens,
                lang_masks=lang_masks,
            )
        if context is None and context_mask is None:
            raise ValueError('Either `lang_tokens/lang_masks` or both '
                             '`context/context_mask` must be provided.')
        if context is None or context_mask is None:
            raise ValueError(
                '`context` and `context_mask` must be provided together.')
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)
        context = context.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(
            device=self.device, dtype=torch.bool, non_blocking=True)

        if proprio is not None and self.vla_head.proprio_encoder is not None:
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)
            context, context_mask = self.vla_head._append_proprio_to_context(
                context=context, context_mask=context_mask, proprio=proprio)
        return context, context_mask

    @torch.no_grad()
    def infer(
        self,
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        lang_tokens: Optional[torch.Tensor] = None,
        lang_masks: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = 'cpu',
        tiled: bool = False,
    ):
        self.eval()
        if action_horizon is None:
            action_horizon = self.action_horizon
        if action_horizon is None:
            raise ValueError(
                '`action_horizon` must be provided or configured on the '
                'model via `action_horizon=`.')
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if (input_image.ndim != 4 or input_image.shape[0] != 1
                or input_image.shape[1] != 3):
            raise ValueError(
                '`input_image` must have shape [1,3,H,W] or [3,H,W], '
                f'got {tuple(input_image.shape)}')
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                '`input_image` spatial dims must be multiples of 16, got '
                f'H={height}, W={width}.')
        if int(num_frames) % 4 != 1:
            raise ValueError(
                f'`num_frames` must satisfy T % 4 == 1, got {num_frames}.')

        input_image = input_image.to(
            device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents(
            input_image, tiled=tiled)
        context, context_mask = self._prepare_inference_context(
            context=context,
            context_mask=context_mask,
            images=input_image,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            proprio=proprio,
        )
        video_latent_shape = self._build_video_latent_shape(
            input_image, num_frames)
        latents_video, pred_action = self.vla_head.predict_video_action(
            first_frame_latents=first_frame_latents,
            context=context,
            context_mask=context_mask,
            action_horizon=int(action_horizon),
            video_latent_shape=video_latent_shape,
            action=action,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
        )
        return {
            'video': self._decode_latents(latents_video, tiled=tiled),
            'action': pred_action,
        }

    # ------------------------------------------------------------------
    # BaseVLA abstract method implementations
    # ------------------------------------------------------------------
    def get_fsdp_wrapping_policy(self) -> Callable:
        from functools import partial

        from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy

        # Wrap the whole head (MoT + video/action experts) as a single FSDP
        # unit. MoT does not call ``expert.forward`` -- it invokes
        # ``expert.pre_dit`` / ``expert.post_dit`` and reads block parameters
        # directly for mixed attention -- so wrapping the experts (or their
        # inner ``DiTBlock``s) would leave their parameters sharded (flat) at
        # access time, because FSDP only all-gathers around a module's
        # ``forward``. Wrapping at ``WAMHead`` makes
        # ``head.forward`` the FSDP boundary, so every parameter the head
        # touches is materialized for the whole step.
        policies = []
        if (self.vlm_backbone is not None
                and hasattr(self.vlm_backbone, 'get_fsdp_wrapping_policy')):
            policies.append(self.vlm_backbone.get_fsdp_wrapping_policy())

        policies.append(
            partial(
                _module_wrap_policy,
                module_classes={WAMHead},
            ))

        if len(policies) == 1:
            return policies[0]
        return partial(_or_policy, policies=policies)

    def load_state_dict(self, state_dict, strict: bool = True, assign=False):
        if not strict:
            return super().load_state_dict(
                state_dict, strict=False, assign=assign)

        incompatible = super().load_state_dict(
            state_dict, strict=False, assign=assign)
        missing_keys = [
            key for key in incompatible.missing_keys
            if not key.startswith('vlm_backbone.')
        ]
        unexpected_keys = [
            key for key in incompatible.unexpected_keys if not (
                self.vlm_backbone is None and key.startswith('vlm_backbone.'))
        ]
        if missing_keys or unexpected_keys:
            messages = []
            if missing_keys:
                messages.append('Missing key(s) in state_dict: {}.'.format(
                    ', '.join(f'"{key}"' for key in missing_keys)))
            if unexpected_keys:
                messages.append('Unexpected key(s) in state_dict: {}.'.format(
                    ', '.join(f'"{key}"' for key in unexpected_keys)))
            raise RuntimeError(
                'Error(s) in loading state_dict for WAMVLA:\n\t' +
                '\n\t'.join(messages))
        return incompatible

    @property
    def config(self):
        from transformers import PretrainedConfig
        cfg = PretrainedConfig()
        cfg.is_encoder_decoder = False
        return cfg
