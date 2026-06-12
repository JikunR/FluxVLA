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
"""Cosmos3-Nano VLA.

Wraps the ``Cosmos3VFMNetwork`` from ``cosmos-framework`` as a FluxVLA
``BaseVLA``, reusing the existing ``ParquetDataset`` + ``FSDPTrainRunner``
infrastructure.

Key differences from DreamZeroVLA / LlavaVLA:
* There is no separate VLM backbone + action head split.  The entire
  Qwen3-VL-8B MoT network handles both understanding and generation.
* Video is encoded by a **frozen** Wan2.2 VAE inside the forward pass (not in
  the dataset pipeline) so that the dataset only needs to ship raw RGB frames.
  The VAE is always run under ``torch.no_grad()`` — gradients do *not* flow
  through the encoder.
* Actions and video latents are jointly denoised via Rectified Flow (flow
  matching) through the same Transformer.
* Sequence packing (``PackedSequence``) is built in ``forward()`` using the
  ``pack_input_sequence`` utility from ``cosmos_framework``.

Checkpoint / weight loading follows the FluxVLA two-step convention:

1. ``__init__`` sets up only lightweight objects (tokenizer, schedulers)
   and stores path references.  No large weight tensors are allocated.
2. ``FSDPTrainRunner.run_setup()`` calls ``self.vla.from_pretrained()``
   **after** the FSDP sharding policy has been resolved but **before** FSDP
   wrapping.  ``from_pretrained`` handles two cases:

   a. *Initial load from a HuggingFace-format cosmos3-nano directory*:
      ``Cosmos3VFMNetwork.from_pretrained(backbone_path)`` + build the
      Wan2.2 VAE from ``vae_path``.
   b. *Resume from a FluxVLA work-dir checkpoint* (``.safetensors`` file or
      directory): load only ``self.net``'s weights via the base-class
      safetensors loader.  The VAE is still constructed from ``vae_path``
      since it is frozen and not saved in training checkpoints.

Data contract (keys expected from the DataLoader / collator):
    images         – ``[B, C, T, H_tiles, W]`` float32 in ``[-1, 1]`` as
                     prepared by ``PrepareVideo`` + ``SimpleNormalizeImages``.
    text_token_ids – ``[B, L]`` int64 token IDs from
                     ``ProcessCosmos3Prompt``.
    actions        – ``[B, T_act, max_action_dim]`` float32, already padded.
    domain_id      – ``[B]`` int64 cosmos3 embodiment domain IDs.
    raw_action_dim – ``[B]`` int64 unpadded action dimensionality (for loss
                     masking).
    sequence_plan  – ``list[SequencePlan]`` (one per sample, not a tensor).
    conditioning_fps – ``[B]`` float32 frame-rate metadata.
"""

from __future__ import annotations
import os
from functools import partial
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
from safetensors.torch import load_file  # resume checkpoint loading

# Short alias for the vendored cosmos3 package.
import fluxvla.models.third_party_models.cosmos3 as _c3  # noqa: F401
from fluxvla.engines import VLAS, initialize_overwatch
from fluxvla.models.vlas.base_vla import BaseVLA

overwatch = initialize_overwatch(__name__)


@VLAS.register_module()
class Cosmos3VLA(BaseVLA):
    """Cosmos3-Nano Vision-Language-Action model.

    Wraps ``Cosmos3VFMNetwork`` (Qwen3-VL-8B MoT) + Wan2.2 VAE from the
    ``cosmos-framework`` package.

    Args:
        pretrained_name_or_path (str): Path to the Cosmos3-Nano HuggingFace
            checkpoint directory (contains ``config.json``,
            ``model.safetensors.index.json``, etc.).  Used on the *first*
            training run to load the backbone weights.
        vae_path (str | None): Path to the Wan2.2 VAE weights file
            (``wan2pt2_vae.pth``).  If ``None``, defaults to
            ``<pretrained_name_or_path>/tokenizer/wan2pt2_vae.pth``.
            The VAE is frozen throughout training and not saved in FluxVLA
            checkpoints, so this must always point to the original backbone
            weights, even when resuming.
        max_action_dim (int): Padded action dimension expected in the batch.
            Defaults to 64 (matches DROIDLeRobot / HUD04 configs).
        action_loss_weight (float): Multiplicative weight applied to the
            action flow-matching loss before summing with the vision loss.
            Defaults to 10.0 (from the Nano model config).
        vision_loss_weight (float): Weight for the vision flow-matching loss.
            Defaults to 1.0.  Set to 0.0 to train action-only (no world
            model).
        num_inference_steps (int): Number of denoising steps during
            ``predict_action``.  Defaults to 20.
        shift (int | Dict[str, int]): Rectified-flow shift parameter.  Can
            be an integer (shared across resolutions) or a dict keyed by
            resolution string, e.g. ``{"256": 3, "480": 5, "720": 10}``.
            Defaults to 3.
        resolution (str): Target video resolution key used to look up the
            shift dict.  Defaults to ``"256"``.
        train_time_action_distribution (str): Noise timestep distribution for
            action tokens.  One of ``"logitnormal"``, ``"uniform"``, or
            ``"waver"``.  Defaults to ``"logitnormal"``.
        train_time_vision_distribution (str): Noise timestep distribution for
            video tokens.  Defaults to ``"waver"``.
        independent_action_schedule (bool): When ``True``, sample action
            timesteps independently from vision timesteps.  Defaults to
            ``True``.
        freeze_vlm_layers (bool): If ``True``, freeze all Transformer
            language-model layers (``net.language_model``) and only train the
            MoT generation experts + projection heads.  Defaults to
            ``False`` (full finetune).
        num_train_timesteps (int): Scheduler resolution for the rectified-
            flow sampler.  Defaults to 1000.
    """

    def __init__(
        self,
        pretrained_name_or_path: str,
        backbone_path: Optional[str] = None,
        vae_path: Optional[str] = None,
        max_action_dim: int = 64,
        action_loss_weight: float = 10.0,
        vision_loss_weight: float = 1.0,
        num_inference_steps: int = 20,
        shift: int | dict = 3,
        resolution: str = '256',
        train_time_action_distribution: str = 'logitnormal',
        train_time_vision_distribution: str = 'waver',
        independent_action_schedule: bool = True,
        freeze_vlm_layers: bool = False,
        num_train_timesteps: int = 1000,
        # Accepted for config compatibility with other VLAs
        name_mapping: Optional[Dict] = None,
        strict_mapping: bool = False,
        **kwargs,
    ) -> None:
        # Bypass the BaseVLA component builders – we manage sub-modules
        # ourselves.  BaseVLA.__init__ requires llm/vision/vlm backbone dicts
        # that don't apply to the MoT architecture.
        nn.Module.__init__(self)

        # ---- paths -------------------------------------------------------
        self.pretrained_name_or_path = pretrained_name_or_path
        # backbone_path: the HF model directory used to build the network
        # architecture.  Defaults to pretrained_name_or_path when that path
        # IS the HF backbone; must be set explicitly when resuming from a
        # FluxVLA work-dir checkpoint (where pretrained_name_or_path points
        # to the checkpoint dir / .safetensors file, not the HF backbone).
        self._backbone_path: str = backbone_path or pretrained_name_or_path
        # _vae_path is resolved once; defaults to <backbone>/tokenizer/
        self._vae_path: str = (
            vae_path or os.path.join(self._backbone_path, 'tokenizer'))
        # BaseVLA.from_pretrained() reads this when doing resume
        self.name_mapping = name_mapping
        self.strict_mapping = strict_mapping

        # ---- hyperparams -------------------------------------------------
        self.max_action_dim = max_action_dim
        self.action_loss_weight = action_loss_weight
        self.vision_loss_weight = vision_loss_weight
        self.num_inference_steps = num_inference_steps
        self.shift = shift
        self.resolution = resolution
        self.train_time_action_distribution = train_time_action_distribution
        self.train_time_vision_distribution = train_time_vision_distribution
        self.independent_action_schedule = independent_action_schedule
        self.freeze_vlm_layers = freeze_vlm_layers
        self.num_train_timesteps = num_train_timesteps

        # Sub-module placeholders – allocated in from_pretrained()
        self.net = None  # Cosmos3VFMNetwork
        self.vae = None  # Wan2pt2VAEInterface
        self.net_config = None
        self.tokenizer = None
        self.special_tokens: Dict = {}

        # Key used by FSDPTrainRunner's checkpoint saver to know which
        # sub-modules to include in the saved state_dict.  VAE is frozen and
        # not saved.
        self.all_module_keys = ['net']

        # ---- lightweight setup -------------------------------------------
        # Tokenizer is small (~100 MB vocab embeddings) and needed by
        # ProcessCosmos3Prompt in the dataset workers, so we initialise
        # it here (no gradient tensors).
        self._setup_tokenizer()
        self._setup_schedulers()

    # ------------------------------------------------------------------
    # Tokenizer (lightweight – loaded at __init__ time)
    # ------------------------------------------------------------------

    def _setup_tokenizer(self) -> None:
        """Load the Qwen3-VL tokenizer and register special tokens."""
        from _c3.data.vfm.sequence_packing import add_special_tokens
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_name_or_path, trust_remote_code=True)
        _tok, special_tokens = add_special_tokens(self.tokenizer)
        self.tokenizer = _tok
        self.special_tokens = special_tokens
        self.special_tokens['eos_token_id'] = self.tokenizer.eos_token_id

    def _setup_schedulers(self) -> None:
        """Create Rectified Flow schedulers for vision and action modalities."""  # noqa: E501
        from _c3.model.vfm.diffusion.rectified_flow import RectifiedFlow
        from _c3.model.vfm.diffusion.samplers.unipc import (  # noqa: E501
            UniPCSampler, UniPCSamplerConfig)

        # Resolve scalar shift for scheduler init (dict-based shift is handled
        # per-sample during forward)
        shift_val = (
            self.shift if isinstance(self.shift, int) else self.shift.get(
                self.resolution, 3))

        # velocity_field=None: forward pass calls net directly
        self.rectified_flow_vision = RectifiedFlow(
            velocity_field=None,
            train_time_distribution=self.train_time_vision_distribution,
            shift=shift_val,
        )
        self.rectified_flow_action = RectifiedFlow(
            velocity_field=None,
            train_time_distribution=self.train_time_action_distribution,
            shift=shift_val,
        )

        unipc_cfg = UniPCSamplerConfig(
            num_train_timesteps=self.num_train_timesteps,
            shift=shift_val,
        )
        self.sampler = UniPCSampler(
            cfg=unipc_cfg,
            tensor_kwargs={
                'device': 'cuda',
                'dtype': torch.float32
            },
        )

    # ------------------------------------------------------------------
    # Weight loading — called by FSDPTrainRunner.run_setup()
    # ------------------------------------------------------------------

    def from_pretrained(self) -> None:
        """Load network and VAE weights.

        Implements the FluxVLA convention: called by ``FSDPTrainRunner.run_setup()``  # noqa: E501
        *after* the model graph is built by ``__init__`` but *before* FSDP
        wrapping.

        Two cases are handled transparently:

        **Initial load** (``self.pretrained_name_or_path`` points to a
        HuggingFace-format cosmos3-nano directory containing
        ``config.json`` + ``model.safetensors.index.json``):
            Uses ``Cosmos3VFMNetwork.from_pretrained()`` (HF API) to load all
            8 B network weights.

        **Resume** (``self.pretrained_name_or_path`` points to a
        ``.safetensors`` file or a FluxVLA work-dir checkpoint directory):
            Builds network architecture from ``self._backbone_path`` (the
            original HF backbone directory), then overlays the resume weights
            via the safetensors loader.  ``backbone_path`` must be set in the
            config when resuming.
            The VAE is still constructed from ``vae_path`` because it is
            frozen and not included in FluxVLA training checkpoints.
        """
        self._load_network()
        self._load_vae()

    def _load_network(self) -> None:
        """Instantiate ``self.net`` from backbone or resume checkpoint."""
        from _c3.model.vfm.mot.cosmos3_vfm_network import (  # noqa: E501
            Cosmos3VFMNetwork, Cosmos3VFMNetworkConfig)

        path = self.pretrained_name_or_path

        # Determine load mode by checking for a HF config.json
        is_hf_backbone = os.path.isdir(path) and os.path.exists(
            os.path.join(path, 'config.json'))
        is_safetensors_file = path.endswith('.safetensors')
        is_resume_dir = os.path.isdir(path) and not is_hf_backbone

        if is_hf_backbone:
            overwatch.info(f'[Cosmos3VLA] Loading backbone from {path}')
            self.net: Cosmos3VFMNetwork = Cosmos3VFMNetwork.from_pretrained(
                path)
            self.net_config: Cosmos3VFMNetworkConfig = self.net.config

        elif is_safetensors_file or is_resume_dir:
            # Resume path: net architecture must be instantiated first from
            # the original HF backbone (self._backbone_path), then weights
            # are overlaid from the checkpoint.
            # self._backbone_path defaults to pretrained_name_or_path for the
            # initial-load case; for resume it must be set via backbone_path.
            backbone = self._backbone_path
            overwatch.info(
                f'[Cosmos3VLA] Resume: building network architecture '
                f'from backbone at {backbone}')
            if not (os.path.isdir(backbone)
                    and os.path.exists(os.path.join(backbone, 'config.json'))):
                raise ValueError(
                    f'[Cosmos3VLA] Resume requires a valid HuggingFace '
                    f'backbone directory with config.json.  '
                    f'Got: {backbone!r}. '
                    f'Set backbone_path in your config to the original HF '
                    f'backbone directory.')
            # Build the empty network from config (no weight download)
            self.net = Cosmos3VFMNetwork.from_pretrained(backbone)
            self.net_config = self.net.config

            # Now overlay the resume weights
            overwatch.info(f'[Cosmos3VLA] Resume: loading weights from {path}')
            if is_safetensors_file:
                resume_weights = load_file(path, device='cpu')
            else:
                resume_weights = {}
                for fname in os.listdir(path):
                    if fname.endswith('.safetensors'):
                        resume_weights.update(
                            load_file(os.path.join(path, fname), device='cpu'))

            # Strip leading 'net.' prefix added by FSDPTrainRunner when saving
            # (it saves self.vla.state_dict() which prefixes sub-modules).
            stripped = {}
            for k, v in resume_weights.items():
                new_k = k[len('net.'):] if k.startswith('net.') else k
                if self.name_mapping:
                    for src, dst in self.name_mapping.items():
                        if src in new_k:
                            new_k = new_k.replace(src, dst)
                stripped[new_k] = v

            missing, unexpected = self.net.load_state_dict(
                stripped, strict=self.strict_mapping)
            if missing:
                overwatch.warning(
                    f'[Cosmos3VLA] Resume: {len(missing)} missing keys')
            if unexpected:
                overwatch.warning(
                    f'[Cosmos3VLA] Resume: {len(unexpected)} unexpected keys'  # noqa: E501
                )

        else:
            raise ValueError(
                f'[Cosmos3VLA] Cannot determine load mode for '
                f'pretrained_name_or_path={path!r}.  '
                f'Expected either a HuggingFace model directory (with '
                f'config.json) or a .safetensors file / work-dir directory.')

    def _load_vae(self) -> None:
        """Instantiate the Wan2.2 VAE tokenizer from ``self._vae_path``."""
        from _c3.model.vfm.tokenizers.wan2pt2_vae_4x16x16 import \
            Wan2pt2VAEInterface

        vae_path = self._vae_path
        overwatch.info(f'[Cosmos3VLA] Loading VAE from {vae_path}')

        # Wan2pt2VAEInterface expects vae_path to be the .pth weight file or
        # the directory containing it.  Resolve to the .pth file if a
        # directory is given.
        if os.path.isdir(vae_path):
            candidates = [
                f for f in os.listdir(vae_path) if f.endswith('.pth')
            ]
            if not candidates:
                raise FileNotFoundError(
                    f'[Cosmos3VLA] No .pth file found in VAE directory: {vae_path}'  # noqa: E501
                )
            vae_file = os.path.join(vae_path, candidates[0])
        else:
            vae_file = vae_path

        self.vae = Wan2pt2VAEInterface(vae_path=vae_file)
        # VAE is frozen: only used for encoding, never trained
        if hasattr(self.vae, 'model') and hasattr(self.vae.model,
                                                  'parameters'):
            for p in self.vae.model.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------------
    # BaseVLA abstract method implementations
    # ------------------------------------------------------------------

    def freeze_backbones(self) -> None:
        """Freeze VLM layers when ``freeze_vlm_layers=True``."""
        if self.freeze_vlm_layers:
            self.net.language_model.requires_grad_(False)
            overwatch.info(
                '[Frozen]    🥶 =>> Cosmos3 VLM (language_model) layers')
        else:
            overwatch.info('[TRAINABLE] 🔥 =>> All Cosmos3 parameters')

        # Log trainable parameters
        overwatch.debug('Trainable parameters:')
        for name, param in self.named_parameters():
            if param.requires_grad:
                overwatch.debug(f'  {name}')

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Wrap at the MoT decoder layer granularity."""
        from _c3.model.vfm.mot.unified_mot import MoTDecoderLayer
        from torch.distributed.fsdp.wrap import _module_wrap_policy

        return partial(_module_wrap_policy, module_classes={MoTDecoderLayer})

    # ------------------------------------------------------------------
    # GenerationMixin stubs (required by BaseVLA / nn.Module subclass chain)
    # ------------------------------------------------------------------

    @property
    def config(self):
        from transformers import PretrainedConfig
        cfg = PretrainedConfig()
        cfg.is_encoder_decoder = False
        return cfg

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_video(self, video: torch.Tensor) -> List[torch.Tensor]:
        """VAE-encode a batch of video tensors to latents.

        Args:
            video: ``[B, C, T, H, W]`` float32 in ``[-1, 1]``.

        Returns:
            List of length *B*, each element ``[C_lat, T_lat, H_lat, W_lat]``.
        """
        B = video.shape[0]
        latents = []
        with torch.no_grad():
            for i in range(B):
                v_i = video[i]  # [C, T, H, W]
                lat_i = self.vae.encode(
                    v_i.unsqueeze(0))  # [1, C_lat, T_lat, H_lat, W_lat]
                latents.append(lat_i.squeeze(0).float().contiguous())
        return latents

    def _sample_timesteps(
        self,
        batch_size: int,
        rectified_flow,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample ``(timesteps, sigmas)`` of shape ``[batch_size, 1]``."""
        max_t = self.num_train_timesteps
        shift_val = (
            self.shift if isinstance(self.shift, int) else self.shift.get(
                self.resolution, 3))
        t_raw = (rectified_flow.sample_train_time(batch_size).to(
            device=device, dtype=torch.float32).unsqueeze(1))  # [B, 1]
        t = 1.0 - t_raw
        s = torch.full((batch_size, 1),
                       shift_val,
                       dtype=torch.float32,
                       device=device)
        timesteps = s * t / (1.0 + (s - 1.0) * t) * max_t  # [B, 1]
        sigmas = timesteps / max_t  # [B, 1]
        return timesteps, sigmas

    def _build_gen_data_clean(
        self,
        x0_vision: List[torch.Tensor],
        x0_action: List[torch.Tensor],
        domain_id: torch.Tensor,
        raw_action_dim: torch.Tensor,
        fps: Optional[torch.Tensor],
        is_image_batch: bool,
    ):
        """Package latents into a ``GenerationDataClean`` dataclass."""
        from _c3.model.vfm.utils.data_and_condition import GenerationDataClean

        # Convert domain_id / raw_action_dim to per-sample tensor lists
        B = len(x0_vision)
        domain_id_list = [domain_id[i].unsqueeze(0) for i in range(B)]
        raw_action_dim_list = [raw_action_dim[i] for i in range(B)
                               ] if raw_action_dim is not None else None

        return GenerationDataClean(
            batch_size=B,
            is_image_batch=is_image_batch,
            x0_tokens_vision=x0_vision,
            fps_vision=fps,
            x0_tokens_action=x0_action,
            fps_action=fps,
            action_domain_id=domain_id_list,
            raw_action_dim=raw_action_dim_list,
        )

    def _add_noise(
        self,
        gen_data_clean,
        packed_sequence,
        sigmas_vision: torch.Tensor,
        sigmas_action: Optional[torch.Tensor],
        precision: torch.dtype,
        device: torch.device,
    ):
        """Apply rectified flow forward process to video latents and actions."""  # noqa: E501
        from _c3.model.vfm.utils.data_and_condition import GenerationDataNoised

        tensor_kwargs = {'device': device, 'dtype': precision}

        # --- Vision ---
        x0_vis = gen_data_clean.x0_tokens_vision  # list[C,T,H,W]
        eps_vis = [torch.randn_like(x.float()) for x in x0_vis]

        noisy_mask_vis = [
            1.0 - cm for cm in packed_sequence.vision.condition_mask
        ]
        sig_vis_per = [
            sigmas_vision[i].view(-1, 1, 1)[:x0_vis[i].shape[1]] *
            noisy_mask_vis[i] for i in range(len(x0_vis))
        ]
        xt_vis, vt_vis = self.rectified_flow_vision.get_interpolation(
            eps_vis, x0_vis, sig_vis_per)
        xt_vis = [x.to(**tensor_kwargs) for x in xt_vis]

        # --- Action ---
        x0_act = gen_data_clean.x0_tokens_action
        eps_act, xt_act, vt_act, sig_act = None, None, None, None
        if x0_act is not None and packed_sequence.action is not None:
            sig_for_act = sigmas_action if sigmas_action is not None else sigmas_vision  # noqa: E501
            n_act = len(packed_sequence.action.condition_mask)
            all_cond = all(
                torch.all(cm == 1)
                for cm in packed_sequence.action.condition_mask)
            if all_cond:
                eps_act = [torch.zeros_like(a.float()) for a in x0_act]
                sig_act = [
                    torch.zeros_like(cm.float())
                    for cm in packed_sequence.action.condition_mask
                ]
                xt_act = [a.to(**tensor_kwargs) for a in x0_act]
                vt_act = [torch.zeros_like(a.float()) for a in x0_act]
            else:
                eps_act = [torch.randn_like(a.float()) for a in x0_act]
                sig_act = [
                    sig_for_act[i].view(-1, 1)[:x0_act[i].shape[0]] *
                    (1.0 - packed_sequence.action.condition_mask[i])
                    for i in range(n_act)
                ]
                xt_act, vt_act = self.rectified_flow_action.get_interpolation(
                    eps_act, x0_act, sig_act)
                xt_act = [x.to(**tensor_kwargs) for x in xt_act]
                # Zero out padded dimensions
                for i, a in enumerate(xt_act):
                    if gen_data_clean.raw_action_dim is not None:
                        rd = int(gen_data_clean.raw_action_dim[i].item())
                        a[:, rd:] = 0

        return GenerationDataNoised(
            batch_size=gen_data_clean.batch_size,
            epsilon_vision=eps_vis,
            xt_tokens_vision=xt_vis,
            vt_target_vision=vt_vis,
            epsilon_action=eps_act,
            xt_tokens_action=xt_act,
            vt_target_action=vt_act,
            raw_action_dim=gen_data_clean.raw_action_dim,
        )

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        text_token_ids: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        domain_id: Optional[torch.Tensor] = None,
        raw_action_dim: Optional[torch.Tensor] = None,
        sequence_plan: Optional[list] = None,
        conditioning_fps: Optional[torch.Tensor] = None,
        # accepted but unused
        task_description: Optional[List[str]] = None,
        states: Optional[torch.Tensor] = None,
        img_masks: Optional[torch.Tensor] = None,
        action_masks: Optional[torch.Tensor] = None,
        embodiment_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass.

        Args:
            images: ``[B, C, T, H, W]`` float32 in ``[-1, 1]`` video tensor.
            text_token_ids: ``[B, L]`` int64 token IDs.
            actions: ``[B, T_act, max_action_dim]`` float32.
            domain_id: ``[B]`` int64 cosmos3 embodiment domain IDs.
            raw_action_dim: ``[B]`` int64 actual (unpadded) action dim.
            sequence_plan: ``list[SequencePlan]``, one per sample.
            conditioning_fps: ``[B]`` float32 frame-rate.

        Returns:
            Dict with keys ``loss``, ``flow_matching_loss_action``,
            ``flow_matching_loss_vision``.
        """
        from _c3.data.vfm.sequence_packing import (  # noqa: F401,E501
            SequencePlan, pack_input_sequence)
        from _c3.model.vfm.algorithm.loss.flow_matching import \
            compute_flow_matching_loss
        from _c3.model.vfm.mot.modeling_utils import has_noisy_tokens

        device = images.device
        B = images.shape[0]
        precision = images.dtype  # bf16 in normal training

        # ------------------------------------------------------------------
        # 1. Encode video to VAE latents
        # ------------------------------------------------------------------
        video_float = images.float()
        x0_vision = self._encode_video(
            video_float)  # list[B] of [C_lat, T_lat, H_lat, W_lat]

        # ------------------------------------------------------------------
        # 2. Prepare action token list  [T_act, max_action_dim] per sample
        # ------------------------------------------------------------------
        x0_action: Optional[List[torch.Tensor]] = None
        if actions is not None:
            x0_action = [actions[i].float() for i in range(B)]

        # ------------------------------------------------------------------
        # 3. Build text token index lists
        # ------------------------------------------------------------------
        if text_token_ids is not None:
            input_text_indexes = [text_token_ids[i].tolist() for i in range(B)]
        else:
            # Empty prompt fallback
            input_text_indexes = [[self.special_tokens['eos_token_id']]] * B

        # ------------------------------------------------------------------
        # 4. Ensure sequence_plan list is available
        # ------------------------------------------------------------------
        if sequence_plan is None:
            # Fallback: all video conditioned, all action generated
            try:
                from _c3.data.vfm.sequence_packing import SequencePlan as SP
                T_vid = x0_vision[0].shape[1] if x0_vision else 1
                sequence_plan = [
                    SP(has_vision=True,
                       has_action=x0_action is not None,
                       condition_frame_indexes_vision=list(range(T_vid)),
                       condition_frame_indexes_action=[]) for _ in range(B)
                ]
            except ImportError:
                raise RuntimeError(
                    'cosmos-framework must be installed for Cosmos3VLA.forward().'  # noqa: E501
                )

        # ------------------------------------------------------------------
        # 5. Build GenerationDataClean
        # ------------------------------------------------------------------
        if domain_id is None:
            domain_id = torch.zeros(B, dtype=torch.long, device=device)
        if raw_action_dim is None and x0_action is not None:
            raw_action_dim = torch.full((B, ),
                                        self.max_action_dim,
                                        dtype=torch.long,
                                        device=device)

        fps = conditioning_fps.float(
        ) if conditioning_fps is not None else None
        gen_data_clean = self._build_gen_data_clean(
            x0_vision=x0_vision,
            x0_action=x0_action if x0_action else [],
            domain_id=domain_id,
            raw_action_dim=raw_action_dim,
            fps=fps,
            is_image_batch=False,
        )

        # ------------------------------------------------------------------
        # 6. Sample training timesteps
        # ------------------------------------------------------------------
        timesteps_vision, sigmas_vision = self._sample_timesteps(
            B, self.rectified_flow_vision, device)  # [B, 1]

        if self.independent_action_schedule and x0_action:
            timesteps_action, sigmas_action = self._sample_timesteps(
                B, self.rectified_flow_action, device)  # [B, 1]
            _independent_action_schedule = True
        else:
            timesteps_action, sigmas_action = timesteps_vision, sigmas_vision
            _independent_action_schedule = False

        # ------------------------------------------------------------------
        # 7. Pack sequence (clean tokens at this point)
        # ------------------------------------------------------------------
        packed_seq = pack_input_sequence(
            sequence_plans=sequence_plan,
            input_text_indexes=input_text_indexes,
            gen_data_clean=gen_data_clean,
            input_timesteps=timesteps_vision.cpu(),
            special_tokens=self.special_tokens,
            latent_patch_size=self.net_config.latent_patch_size,
            position_embedding_type=self.net_config.position_embedding_type,
            temporal_compression_factor=self.vae.temporal_compression_factor,
            action_dim=self.max_action_dim,
            base_fps=float(getattr(self.net_config, 'base_fps', 24)),
            video_temporal_causal=getattr(self.net_config,
                                          'video_temporal_causal', False),
            enable_fps_modulation=getattr(self.net_config,
                                          'enable_fps_modulation', False),
        )

        # Overwrite action timesteps if independent schedule
        if _independent_action_schedule and packed_seq.action is not None:  # noqa: E501
            act_has_noisy = any(
                nfi.numel() > 0
                for nfi in packed_seq.action.noisy_frame_indexes)
            if act_has_noisy:
                sample_ts = timesteps_action.squeeze(1).cpu()  # [B]
                packed_seq.action.timesteps = torch.cat([
                    sample_ts[i:i + 1].expand(nfi.numel()) for i, nfi in
                    enumerate(packed_seq.action.noisy_frame_indexes)
                ]).float()

        # ------------------------------------------------------------------
        # 8. Add noise (forward diffusion process)
        # ------------------------------------------------------------------
        gen_data_noised = self._add_noise(
            gen_data_clean=gen_data_clean,
            packed_sequence=packed_seq,
            sigmas_vision=sigmas_vision,
            sigmas_action=sigmas_action
            if _independent_action_schedule else None,
            precision=precision,
            device=device,
        )

        # Replace clean tokens with noisy tokens in packed_seq
        if packed_seq.vision is not None:
            packed_seq.vision.tokens = gen_data_noised.xt_tokens_vision
        if (packed_seq.action is not None
                and gen_data_noised.xt_tokens_action is not None):
            all_cond = all(
                torch.all(cm == 1) for cm in packed_seq.action.condition_mask)
            if not all_cond:
                packed_seq.action.tokens = gen_data_noised.xt_tokens_action

        # ------------------------------------------------------------------
        # 9. Network forward pass
        # ------------------------------------------------------------------
        packed_seq.to_cuda()
        out_net = self.net(
            packed_seq=packed_seq,
            fps_vision=gen_data_clean.fps_vision,
            fps_action=gen_data_clean.fps_action,
        )

        # ------------------------------------------------------------------
        # 10. Compute losses
        # ------------------------------------------------------------------
        tensor_kwargs_fp32 = {'device': device, 'dtype': torch.float32}
        total_loss = torch.tensor(0.0, **tensor_kwargs_fp32)
        losses: Dict[str, torch.Tensor] = {}

        # Vision loss
        if packed_seq.vision is not None and out_net.get('preds_vision'):
            fm_loss_vis, _ = compute_flow_matching_loss(
                pred=out_net['preds_vision'],
                target=gen_data_noised.vt_target_vision,
                condition_mask=packed_seq.vision.condition_mask,
                timesteps=timesteps_vision,
                has_valid_tokens=has_noisy_tokens(packed_seq.vision),
                rectified_flow=self.rectified_flow_vision,
                tensor_kwargs_fp32=tensor_kwargs_fp32,
            )
            total_loss = total_loss + fm_loss_vis * self.vision_loss_weight
            losses['flow_matching_loss_vision'] = fm_loss_vis.detach()

        # Action loss
        if (packed_seq.action is not None and out_net.get('preds_action')
                and gen_data_noised.vt_target_action is not None):
            fm_loss_act, _ = compute_flow_matching_loss(
                pred=out_net['preds_action'],
                target=gen_data_noised.vt_target_action,
                condition_mask=packed_seq.action.condition_mask,
                timesteps=timesteps_action,
                has_valid_tokens=has_noisy_tokens(packed_seq.action),
                rectified_flow=self.rectified_flow_action,
                tensor_kwargs_fp32=tensor_kwargs_fp32,
                raw_action_dim=packed_seq.action.raw_action_dim,
            )
            total_loss = total_loss + fm_loss_act * self.action_loss_weight
            losses['flow_matching_loss_action'] = fm_loss_act.detach()
        elif packed_seq.action is not None and out_net.get('preds_action'):
            # Keep action params in the compute graph (avoid FSDP hang)
            dummy = 0.0 * sum(p.sum() for p in out_net['preds_action'])
            total_loss = total_loss + dummy
            losses['flow_matching_loss_action'] = torch.tensor(
                0.0, **tensor_kwargs_fp32)

        losses['loss'] = total_loss
        return losses

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_action(
        self,
        images: torch.Tensor,
        text_token_ids: torch.Tensor,
        domain_id: Optional[torch.Tensor] = None,
        raw_action_dim: Optional[torch.Tensor] = None,
        action_horizon: int = 16,
        conditioning_fps: Optional[torch.Tensor] = None,
        # accepted but unused
        lang_tokens: Optional[torch.Tensor] = None,
        lang_masks: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
        embodiment_ids: Optional[torch.Tensor] = None,
        reset_history: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Run policy inference (all video conditioned, action denoised from noise).  # noqa: E501

        Args:
            images: ``[B, C, T, H, W]`` float32 video in ``[-1, 1]``.
            text_token_ids: ``[B, L]`` int64 prompt token IDs.
            domain_id: ``[B]`` int64 cosmos3 domain ID.
            raw_action_dim: ``[B]`` int64 true (unpadded) action dim.
            action_horizon: Number of action time steps to predict.

        Returns:
            ``[B, T_act, raw_action_dim]`` float32 predicted actions.
        """
        from _c3.data.vfm.sequence_packing import SequencePlan  # noqa: E501
        from _c3.data.vfm.sequence_packing import pack_input_sequence
        from _c3.model.vfm.utils.data_and_condition import GenerationDataClean

        device = images.device
        B = images.shape[0]
        T_vid = images.shape[2]

        # Encode video
        x0_vision = self._encode_video(images.float())

        # Text tokens
        input_text_indexes = [text_token_ids[i].tolist() for i in range(B)]

        # Build SequencePlan (policy mode: all video frames conditioned, actions generated)  # noqa: E501
        seq_plans = [
            SequencePlan(
                has_vision=True,
                has_action=True,
                has_sound=False,
                condition_frame_indexes_vision=list(range(T_vid)),
                condition_frame_indexes_action=[],
            ) for _ in range(B)
        ]

        if domain_id is None:
            domain_id = torch.zeros(B, dtype=torch.long, device=device)
        if raw_action_dim is None:
            raw_action_dim = torch.full((B, ),
                                        self.max_action_dim,
                                        dtype=torch.long,
                                        device=device)

        domain_id_list = [domain_id[i].unsqueeze(0) for i in range(B)]
        raw_action_dim_list = [raw_action_dim[i] for i in range(B)]

        # Initialize action from pure noise
        fps = conditioning_fps.float(
        ) if conditioning_fps is not None else None
        x0_action_noise = [
            torch.randn(
                action_horizon,
                self.max_action_dim,
                device='cpu',
                dtype=torch.float32) for _ in range(B)
        ]

        gen_data_clean = GenerationDataClean(
            batch_size=B,
            is_image_batch=False,
            x0_tokens_vision=x0_vision,
            fps_vision=fps,
            x0_tokens_action=x0_action_noise,
            fps_action=fps,
            action_domain_id=domain_id_list,
            raw_action_dim=raw_action_dim_list,
        )

        # Use fixed timestep schedule (t=1000 → 0)
        timesteps_all = self.sampler.get_timesteps(self.num_inference_steps)

        current_actions = x0_action_noise  # [T_act, max_action_dim] per sample, on CPU  # noqa: E501

        for step_idx, t in enumerate(timesteps_all):
            t_tensor = torch.tensor([[t]], dtype=torch.float32)  # [1, 1]
            t_batch = t_tensor.expand(B, 1)  # [B, 1]

            # Update gen_data_clean with current (noisy) actions
            gen_data_clean.x0_tokens_action = current_actions

            packed_seq = pack_input_sequence(
                sequence_plans=seq_plans,
                input_text_indexes=input_text_indexes,
                gen_data_clean=gen_data_clean,
                input_timesteps=t_batch,
                special_tokens=self.special_tokens,
                latent_patch_size=self.net_config.latent_patch_size,
                position_embedding_type=self.net_config.
                position_embedding_type,
                temporal_compression_factor=self.vae.
                temporal_compression_factor,
                action_dim=self.max_action_dim,
                base_fps=float(getattr(self.net_config, 'base_fps', 24)),
                video_temporal_causal=getattr(self.net_config,
                                              'video_temporal_causal', False),
                enable_fps_modulation=getattr(self.net_config,
                                              'enable_fps_modulation', False),
            )
            packed_seq.to_cuda()

            out = self.net(
                packed_seq=packed_seq,
                fps_vision=gen_data_clean.fps_vision,
                fps_action=gen_data_clean.fps_action,
            )

            if out.get('preds_action'):
                # Update current_actions using sampler step
                # preds_action: list[B] of [T_act, max_action_dim]
                for i in range(B):
                    pred_v = out['preds_action'][i].cpu().float()
                    # x_{t-dt} = x_t - v_pred * (sigma_current - sigma_next)
                    sigma_curr = t / self.num_train_timesteps
                    if step_idx + 1 < len(timesteps_all):
                        sigma_next = timesteps_all[
                            step_idx + 1] / self.num_train_timesteps
                    else:
                        sigma_next = 0.0
                    dt = sigma_curr - sigma_next
                    current_actions[i] = current_actions[i] - pred_v * dt

        # Stack and slice to per-sample true action dims
        actions_batch = torch.stack(
            current_actions, dim=0)  # [B, T_act, max_action_dim]
        # Slice each sample to its own raw_action_dim to avoid cross-sample
        # truncation when batch contains mixed embodiments.
        rad_list = [int(raw_action_dim[i].item()) for i in range(B)]
        if len(set(rad_list)) == 1:
            # All samples share the same action dim — single slice is safe
            return actions_batch[:, :, :rad_list[0]]
        # Mixed dims: return a list of [T_act, rad_i] tensors padded to max
        max_rad = max(rad_list)
        result = torch.zeros(
            B, actions_batch.shape[1], max_rad, dtype=actions_batch.dtype)
        for i, rad in enumerate(rad_list):
            result[i, :, :rad] = actions_batch[i, :, :rad]
        return result

    # ------------------------------------------------------------------
    # GenerationMixin required methods (no-op stubs)
    # ------------------------------------------------------------------

    def _reorder_cache(self, past_key_values, beam_idx):
        return past_key_values
