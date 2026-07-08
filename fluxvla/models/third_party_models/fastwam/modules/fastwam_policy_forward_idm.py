from typing import Any, Optional

import torch
import torch.nn.functional as F

from .fastwam_idm import FastWAMIDM


class FastWAMPolicyForwardIDM(FastWAMIDM):
    """Cosmos-style random-mode training for policy, forward dynamics, and IDM."""

    valid_training_modes = ("forward", "idm", "policy")

    def __init__(
        self,
        *args,
        loss_lambda_forward_video: float = 1.0,
        loss_lambda_idm_action: float = 1.0,
        loss_lambda_policy_action: float = 1.0,
        mode_probs: Optional[dict[str, float]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_lambda_forward_video = float(loss_lambda_forward_video)
        self.loss_lambda_idm_action = float(loss_lambda_idm_action)
        self.loss_lambda_policy_action = float(loss_lambda_policy_action)
        self.mode_probs = self._normalize_mode_probs(mode_probs)

    @classmethod
    def from_wan22_pretrained(cls, **kwargs):
        loss_lambda_forward_video = float(kwargs.pop("loss_lambda_forward_video", 1.0))
        loss_lambda_idm_action = float(kwargs.pop("loss_lambda_idm_action", 1.0))
        loss_lambda_policy_action = float(kwargs.pop("loss_lambda_policy_action", 1.0))
        mode_probs = kwargs.pop("mode_probs", None)

        video_dit_config = kwargs.get("video_dit_config", None)
        if not isinstance(video_dit_config, dict):
            raise ValueError(
                "`video_dit_config` must be provided as dict for FastWAMPolicyForwardIDM."
            )
        if bool(video_dit_config.get("action_conditioned", False)):
            raise ValueError(
                "FastWAMPolicyForwardIDM requires `video_dit_config['action_conditioned']=false`; "
                "actions are provided through ActionDiT tokens in MoT attention."
            )
        if str(video_dit_config.get("video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "FastWAMPolicyForwardIDM requires "
                "`video_dit_config['video_attention_mask_mode']='first_frame_causal'`."
            )
        if not bool(video_dit_config.get("seperated_timestep", False)):
            raise ValueError(
                "FastWAMPolicyForwardIDM requires `video_dit_config['seperated_timestep']=true`."
            )
        if not bool(video_dit_config.get("fuse_vae_embedding_in_latents", False)):
            raise ValueError(
                "FastWAMPolicyForwardIDM requires "
                "`video_dit_config['fuse_vae_embedding_in_latents']=true`."
            )

        model = super().from_wan22_pretrained(**kwargs)
        model.loss_lambda_forward_video = loss_lambda_forward_video
        model.loss_lambda_idm_action = loss_lambda_idm_action
        model.loss_lambda_policy_action = loss_lambda_policy_action
        model.mode_probs = model._normalize_mode_probs(mode_probs)
        return model

    @classmethod
    def _normalize_mode_probs(cls, mode_probs: Optional[dict[str, float]]) -> dict[str, float]:
        if mode_probs is None:
            mode_probs = {mode: 1.0 for mode in cls.valid_training_modes}
        unknown = set(mode_probs) - set(cls.valid_training_modes)
        if unknown:
            raise ValueError(f"Unknown FastWAMPolicyForwardIDM mode(s): {sorted(unknown)}")

        probs = {mode: float(mode_probs.get(mode, 0.0)) for mode in cls.valid_training_modes}
        if any(value < 0.0 for value in probs.values()):
            raise ValueError(f"`mode_probs` must be non-negative, got {probs}")
        total = sum(probs.values())
        if total <= 0.0:
            raise ValueError("At least one `mode_probs` entry must be positive.")
        return {mode: value / total for mode, value in probs.items()}

    def _sample_training_mode(self) -> str:
        probs = torch.tensor(
            [self.mode_probs[mode] for mode in self.valid_training_modes],
            device=self.device,
            dtype=torch.float32,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            backend = str(torch.distributed.get_backend()).lower()
            idx_device = torch.device("cpu") if backend == "gloo" else self.device
            idx_tensor = torch.empty((), device=idx_device, dtype=torch.long)
            if torch.distributed.get_rank() == 0:
                sampled = torch.multinomial(probs, num_samples=1).to(device=idx_device, dtype=torch.long)
                idx_tensor.copy_(sampled[0])
            torch.distributed.broadcast(idx_tensor, src=0)
            return self.valid_training_modes[int(idx_tensor.cpu().item())]

        idx = int(torch.multinomial(probs, num_samples=1).item())
        return self.valid_training_modes[idx]

    @torch.no_grad()
    def _build_forward_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        mask[video_seq_len:, video_seq_len:] = True
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[first_frame_tokens:video_seq_len, video_seq_len:] = True
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
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        mask[video_seq_len:, video_seq_len:] = True
        mask[video_seq_len:, :video_seq_len] = True
        return mask

    def _compute_action_loss(
        self,
        pred_action: torch.Tensor,
        target_action: torch.Tensor,
        timestep_action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        action_loss_token = F.mse_loss(
            pred_action.float(),
            target_action.float(),
            reduction="none",
        ).mean(dim=2)
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device,
            dtype=action_loss_per_sample.dtype,
        )
        return (action_loss_per_sample * action_weight).mean()

    def _compute_forward_video_loss(self, inputs: dict[str, Any]) -> torch.Tensor:
        input_latents = inputs["input_latents"]
        action = inputs["action"]
        batch_size = int(input_latents.shape[0])

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)
        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        timestep_action = torch.zeros((batch_size,), dtype=action.dtype, device=self.device)
        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=inputs["context"],
            context_mask=inputs["context_mask"],
            action=None,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=action,
            timestep=timestep_action,
            context=inputs["context"],
            context_mask=inputs["context_mask"],
        )
        if video_pre["t_mod"].ndim != 4:
            raise ValueError(
                "Forward branch requires token-wise video `t_mod`; "
                "ensure `seperated_timestep=true` and `fuse_vae_embedding_in_latents=true`."
            )

        attention_mask = self._build_forward_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        zero_anchor = self.action_expert.post_dit(tokens_out["action"], action_pre).float().sum() * 0.0

        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=inputs["image_is_pad"],
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device,
            dtype=loss_video_per_sample.dtype,
        )
        return (loss_video_per_sample * video_weight).mean() + zero_anchor

    def _compute_idm_action_loss(self, inputs: dict[str, Any]) -> torch.Tensor:
        input_latents = inputs["input_latents"]
        action = inputs["action"]
        batch_size = int(input_latents.shape[0])

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        cond_noise_mask = torch.rand((batch_size,), device=self.device) < float(self.video_cond_noise_prob)
        timestep_video_cond = torch.zeros((batch_size,), dtype=input_latents.dtype, device=self.device)
        latents_cond = input_latents
        if bool(cond_noise_mask.any()):
            timestep_video_cond_sampled = self.train_video_scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=input_latents.dtype,
            )
            timestep_video_cond = torch.where(
                cond_noise_mask,
                timestep_video_cond_sampled,
                timestep_video_cond,
            )
            noise_video_cond = torch.randn_like(input_latents)
            latents_cond_noisy = self.train_video_scheduler.add_noise(
                input_latents,
                noise_video_cond,
                timestep_video_cond_sampled,
            )
            cond_noise_selector = cond_noise_mask.view(batch_size, 1, 1, 1, 1)
            latents_cond = torch.where(cond_noise_selector, latents_cond_noisy, input_latents)
        if inputs["first_frame_latents"] is not None:
            latents_cond = latents_cond.clone()
            latents_cond[:, :, 0:1] = inputs["first_frame_latents"]

        video_pre_cond = self.video_expert.pre_dit(
            x=latents_cond,
            timestep=timestep_video_cond,
            context=inputs["context"],
            context_mask=inputs["context_mask"],
            action=None,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=inputs["context"],
            context_mask=inputs["context_mask"],
        )
        if video_pre_cond["t_mod"].ndim != 4:
            raise ValueError(
                "IDM branch requires token-wise video `t_mod`; "
                "ensure `seperated_timestep=true` and `fuse_vae_embedding_in_latents=true`."
            )

        attention_mask = self._build_policy_attention_mask(
            video_seq_len=video_pre_cond["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre_cond["meta"]["tokens_per_frame"]),
            device=video_pre_cond["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre_cond["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre_cond["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre_cond["context"],
                    "mask": video_pre_cond["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre_cond["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        zero_anchor = self.video_expert.post_dit(tokens_out["video"], video_pre_cond).float().sum() * 0.0
        return self._compute_action_loss(
            pred_action=pred_action,
            target_action=target_action,
            timestep_action=timestep_action,
            action_is_pad=inputs["action_is_pad"],
        ) + zero_anchor

    def _compute_policy_action_loss(self, inputs: dict[str, Any]) -> torch.Tensor:
        first_frame_latents = inputs["first_frame_latents"]
        if first_frame_latents is None:
            first_frame_latents = inputs["input_latents"][:, :, 0:1]

        action = inputs["action"]
        batch_size = int(action.shape[0])
        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        timestep_video = torch.zeros((batch_size,), dtype=first_frame_latents.dtype, device=self.device)
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=inputs["context"],
            context_mask=inputs["context_mask"],
            action=None,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=inputs["context"],
            context_mask=inputs["context_mask"],
        )

        attention_mask = self._build_policy_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        zero_anchor = self.video_expert.post_dit(tokens_out["video"], video_pre).float().sum() * 0.0
        return self._compute_action_loss(
            pred_action=pred_action,
            target_action=target_action,
            timestep_action=timestep_action,
            action_is_pad=inputs["action_is_pad"],
        ) + zero_anchor

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        mode = self._sample_training_mode()

        if mode == "forward":
            loss_branch = self._compute_forward_video_loss(inputs)
            loss_total = self.loss_lambda_forward_video * loss_branch
        elif mode == "idm":
            loss_branch = self._compute_idm_action_loss(inputs)
            loss_total = self.loss_lambda_idm_action * loss_branch
        elif mode == "policy":
            loss_branch = self._compute_policy_action_loss(inputs)
            loss_total = self.loss_lambda_policy_action * loss_branch
        else:
            raise ValueError(f"Unsupported training mode: {mode}")

        active_loss = float(loss_total.detach().item())
        loss_dict = {
            "loss_forward_video": active_loss if mode == "forward" else 0.0,
            "loss_idm_action": active_loss if mode == "idm" else 0.0,
            "loss_policy_action": active_loss if mode == "policy" else 0.0,
            "mode_forward": 1.0 if mode == "forward" else 0.0,
            "mode_idm": 1.0 if mode == "idm" else 0.0,
            "mode_policy": 1.0 if mode == "policy" else 0.0,
        }
        return loss_total, loss_dict

    def _prepare_inference_inputs(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: Optional[int],
        action_horizon: Optional[int],
        action: Optional[torch.Tensor],
        proprio: Optional[torch.Tensor],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
        tiled: bool,
    ) -> dict[str, Any]:
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        latent_shape = None
        if num_video_frames is not None:
            checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
            if (checked_h, checked_w) != (height, width):
                raise ValueError(
                    "`input_image` must be resized before infer, expected multiples of 16 "
                    f"but got HxW=({height},{width})"
                )
            if checked_t != num_video_frames:
                raise ValueError(f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}")
            latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
            latent_h = height // self.vae.upsampling_factor
            latent_w = width // self.vae.upsampling_factor
            latent_shape = (latent_t, latent_h, latent_w)

        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1:
                raise ValueError(f"`action` must have shape [1,T,D] or [T,D], got {tuple(action.shape)}")
            if action.shape[2] != self.action_expert.action_dim:
                raise ValueError(
                    f"`action` last dim must be {self.action_expert.action_dim}, got {action.shape[2]}"
                )
            if action_horizon is not None and action.shape[1] != action_horizon:
                raise ValueError(
                    f"`action` horizon must match `action_horizon`, got {action.shape[1]} and {action_horizon}"
                )
            action_horizon = int(action.shape[1])
            action = action.to(device=self.device, dtype=self.torch_dtype)
        elif action_horizon is None:
            raise ValueError("`action_horizon` is required when `action` is not provided.")

        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")
        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        return {
            "action": action,
            "action_horizon": int(action_horizon),
            "context": context,
            "context_mask": context_mask,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)),
            "latent_shape": latent_shape,
        }

    @torch.no_grad()
    def _predict_forward_video_noise(
        self,
        latents_video: torch.Tensor,
        action: torch.Tensor,
        timestep_video: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> torch.Tensor:
        timestep_action = torch.zeros((action.shape[0],), dtype=action.dtype, device=self.device)
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        attention_mask = self._build_forward_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        return self.video_expert.post_dit(tokens_out["video"], video_pre)

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        del negative_prompt, text_cfg_scale
        self.eval()
        prepared = self._prepare_inference_inputs(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_video_frames,
            action_horizon=action_horizon,
            action=None,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            tiled=tiled,
        )

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, prepared["action_horizon"], self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        timestep_video = torch.zeros(
            (prepared["first_frame_latents"].shape[0],),
            dtype=prepared["first_frame_latents"].dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=prepared["first_frame_latents"],
            timestep=timestep_video,
            context=prepared["context"],
            context_mask=prepared["context_mask"],
            action=None,
            fuse_vae_embedding_in_latents=prepared["fuse_vae_embedding_in_latents"],
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_policy_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
            pred_action = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=prepared["context"],
                context_mask=prepared["context_mask"],
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        return {"action": latents_action[0].detach().to(device="cpu", dtype=torch.float32)}

    @torch.no_grad()
    def infer_forward_video(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action: torch.Tensor,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        del negative_prompt, text_cfg_scale
        self.eval()
        prepared = self._prepare_inference_inputs(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_video_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            tiled=tiled,
        )
        latent_t, latent_h, latent_w = prepared["latent_shape"]
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_video[:, :, 0:1] = prepared["first_frame_latents"].clone()

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video in zip(infer_timesteps_video, infer_deltas_video):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            pred_video = self._predict_forward_video_noise(
                latents_video=latents_video,
                action=prepared["action"],
                timestep_video=timestep_video,
                context=prepared["context"],
                context_mask=prepared["context_mask"],
                fuse_vae_embedding_in_latents=prepared["fuse_vae_embedding_in_latents"],
            )
            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_video[:, :, 0:1] = prepared["first_frame_latents"].clone()

        return {"video": self._decode_latents(latents_video, tiled=tiled)}

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: Optional[int] = None,
        action: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = True,
    ) -> dict[str, Any]:
        del test_action_with_infer_action
        if action is None:
            if action_horizon is None:
                raise ValueError("`action_horizon` is required when `action` is not provided.")
            action_out = self.infer_action(
                prompt=prompt,
                input_image=input_image.clone(),
                action_horizon=action_horizon,
                num_video_frames=num_video_frames,
                proprio=proprio.clone() if proprio is not None else None,
                context=context.clone() if context is not None else None,
                context_mask=context_mask.clone() if context_mask is not None else None,
                negative_prompt=negative_prompt,
                text_cfg_scale=text_cfg_scale,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
            )["action"]
            action_for_forward = action_out.unsqueeze(0).to(device=self.device, dtype=self.torch_dtype)
        else:
            action_for_forward = action
            if action_for_forward.ndim == 3:
                action_out = action_for_forward[0].detach().to(device="cpu", dtype=torch.float32)
            else:
                action_out = action_for_forward.detach().to(device="cpu", dtype=torch.float32)

        video_out = self.infer_forward_video(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_video_frames,
            action=action_for_forward,
            action_horizon=action_horizon,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )["video"]
        return {"video": video_out, "action": action_out}
