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
from .state_to_action_decoder import StateToActionDecoder
from .wam_head import WAMHead

_ACTION_DECODER_BUILDERS = {
    'StateToActionDecoder': StateToActionDecoder,
}


@HEADS.register_module()
class WAMStateChunkHead(WAMHead):
    """WAM variant that predicts state chunks and controller actions.

    The inherited MoT branch still uses the parent's internal ``action`` slot,
    but semantically that slot is the state/action chunk expert in this head.

    Two operating modes are supported:

    * ``joint_state_action=False`` (legacy): the MoT slot predicts state
      chunks only, and a small teacher-forced decoder then maps GT/predicted
      state chunks to action chunks.
    * ``joint_state_action=True``: the MoT slot predicts one concatenated
      ``[state_chunks | actions]`` chunk with a single diffusion expert, so
      state and action are supervised jointly and the separate action decoder
      is not used.
    """

    def __init__(
        self,
        state_expert: Optional[Mapping[str, Any] | nn.Module] = None,
        action_decoder: Optional[Mapping[str, Any] | nn.Module] = None,
        action_dim: Optional[int] = None,
        joint_state_action: bool = False,
        loss: Optional[Dict[str, Any]] = None,
        loss_lambda_state_to_action: float = 1.0,
        device: str = 'cpu',
        torch_dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> None:
        if state_expert is not None:
            if 'action_expert' in kwargs:
                raise ValueError(
                    'Pass only one of `state_expert` and `action_expert`.')
            kwargs['action_expert'] = state_expert

        joint_state_action = bool(joint_state_action)
        loss_cfg = dict(loss or {})
        idm_state_key = ('lambda_idm_state_action'
                         if joint_state_action else 'lambda_idm_state')
        policy_state_key = ('lambda_policy_state_action'
                            if joint_state_action else 'lambda_policy_state')
        joint_state_key = ('lambda_joint_state_action'
                           if joint_state_action else 'lambda_joint_state')
        loss_cfg.setdefault('lambda_idm_action',
                            loss_cfg.get(idm_state_key, 1.0))
        loss_cfg.setdefault('lambda_policy_action',
                            loss_cfg.get(policy_state_key, 1.0))
        loss_cfg.setdefault('lambda_joint_action',
                            loss_cfg.get(joint_state_key, 1.0))
        self.loss_lambda_state_to_action = float(
            loss_cfg.get('lambda_state_to_action',
                         loss_lambda_state_to_action))

        super().__init__(
            loss=loss_cfg,
            device=device,
            torch_dtype=torch_dtype,
            **kwargs,
        )

        self.controller_action_dim = (None if action_dim is None else
                                      int(action_dim))
        self.action_decoder = self._build_action_decoder(
            action_decoder,
            action_dim=self.controller_action_dim,
            device=device,
            torch_dtype=torch_dtype,
        )
        if self.action_decoder is not None:
            self.controller_action_dim = self.action_decoder.action_dim
        self.joint_state_action = joint_state_action
        if self.joint_state_action:
            if self.action_decoder is not None:
                raise ValueError(
                    '`joint_state_action` cannot be combined with an '
                    '`action_decoder`; the joint MoT slot already predicts '
                    'state chunks and actions together.')
            if self.controller_action_dim is None:
                raise ValueError(
                    '`action_dim` is required for joint state-action '
                    'prediction.')
            self.state_chunk_dim = (
                self.action_expert.action_dim - self.controller_action_dim)
            if self.state_chunk_dim <= 0:
                raise ValueError(
                    'Joint state-action `state_expert.action_dim` must be '
                    'larger than `action_dim`; got '
                    f'{self.action_expert.action_dim} vs '
                    f'{self.controller_action_dim}.')
        else:
            self.state_chunk_dim = self.action_expert.action_dim

    def _build_action_decoder(
        self,
        component,
        *,
        action_dim: Optional[int],
        device: str,
        torch_dtype: torch.dtype,
    ):
        if component is None:
            return None
        if isinstance(component, nn.Module):
            return component
        if not isinstance(component, Mapping):
            raise TypeError('`action_decoder` must be a dict or '
                            f'nn.Module, got {type(component).__name__}.')
        cfg = dict(component)
        component_type = cfg.pop('type', None)
        if component_type is None:
            raise ValueError('`action_decoder.type` is required.')
        if isinstance(component_type, str):
            builder = _ACTION_DECODER_BUILDERS.get(component_type)
            if builder is None:
                raise KeyError(f'{component_type!r} is not a built-in '
                               'action decoder component.')
        elif callable(component_type):
            builder = component_type
        else:
            raise TypeError('`action_decoder.type` must be a string or '
                            f'callable, got {type(component_type).__name__}.')

        cfg.setdefault('state_dim', self.action_expert.action_dim)
        if action_dim is not None:
            cfg.setdefault('action_dim', int(action_dim))
        if 'action_dim' not in cfg:
            raise ValueError('`action_dim` is required for '
                             '`action_decoder`.')
        cfg.setdefault('device', device)
        cfg.setdefault('torch_dtype', torch_dtype)
        return builder(**cfg)

    @staticmethod
    def _rename_state_loss_keys(ret: Dict[str, torch.Tensor],
                                joint_state_action: bool = False):
        if joint_state_action:
            if 'loss_idm_action' in ret:
                ret['loss_idm_state_action'] = ret.pop('loss_idm_action')
            if 'loss_policy_action' in ret:
                ret['loss_policy_state_action'] = ret.pop('loss_policy_action')
            if 'loss_joint_action' in ret:
                ret['loss_joint_state_action'] = ret.pop('loss_joint_action')
            return
        if 'loss_idm_action' in ret:
            ret['loss_idm_state'] = ret.pop('loss_idm_action')
        if 'loss_policy_action' in ret:
            ret['loss_policy_state'] = ret.pop('loss_policy_action')
        if 'loss_joint_action' in ret:
            ret['loss_joint_state'] = ret.pop('loss_joint_action')

    @staticmethod
    def _masked_chunk_mse(
        pred: torch.Tensor,
        target: torch.Tensor,
        is_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        loss_per_token = F.mse_loss(
            pred.float(),
            target.float(),
            reduction='none',
        ).mean(dim=-1)
        if is_pad is None:
            return loss_per_token.mean()

        valid = (~is_pad).to(
            device=loss_per_token.device, dtype=loss_per_token.dtype)
        valid_count = valid.sum(dim=1).clamp(min=1.0)
        return ((loss_per_token * valid).sum(dim=1) / valid_count).mean()

    def _to_model_dtype(self, tensor: torch.Tensor,
                        device: torch.device) -> torch.Tensor:
        return tensor.to(
            device=device,
            dtype=self.torch_dtype,
            non_blocking=True,
        )

    @staticmethod
    def _to_bool_mask(mask: Optional[torch.Tensor],
                      device: torch.device) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        return mask.to(device=device, dtype=torch.bool, non_blocking=True)

    def _compute_state_to_action_loss(
        self,
        current_state: torch.Tensor,
        state_chunks: torch.Tensor,
        target_action: torch.Tensor,
        state_chunk_is_pad: Optional[torch.Tensor],
        target_action_is_pad: Optional[torch.Tensor],
        embodiment_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.action_decoder is None:
            return target_action.new_zeros(())

        state_masks = None
        if state_chunk_is_pad is not None:
            state_masks = ~state_chunk_is_pad

        pred_action = self.action_decoder(
            current_state=current_state,
            state_chunks=state_chunks,
            state_masks=state_masks,
            embodiment_ids=embodiment_ids,
        )
        if pred_action.shape != target_action.shape:
            raise ValueError('state_to_action output shape mismatch: '
                             f'pred={tuple(pred_action.shape)} vs '
                             f'target={tuple(target_action.shape)}')

        is_pad = target_action_is_pad
        if state_chunk_is_pad is not None:
            is_pad = (
                state_chunk_is_pad if is_pad is None else
                (is_pad | state_chunk_is_pad))
        return self._masked_chunk_mse(
            pred=pred_action,
            target=target_action,
            is_pad=is_pad,
        )

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
        state_chunks: Optional[torch.Tensor] = None,
        state_chunk_is_pad: Optional[torch.Tensor] = None,
        embodiment_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if state_chunks is None:
            raise ValueError(
                '`state_chunks` is required for WAMStateChunkHead.')

        state_target = state_chunks
        state_target_is_pad = state_chunk_is_pad
        action_target = action
        action_target_is_pad = action_is_pad
        device = input_latents.device

        state_target = self._to_model_dtype(state_target, device)
        state_target_is_pad = self._to_bool_mask(state_target_is_pad, device)

        if self.joint_state_action:
            if action_target is None:
                raise ValueError(
                    '`action` is required for joint state-action prediction.')
            action_target = self._to_model_dtype(action_target, device)
            action_target_is_pad = self._to_bool_mask(action_target_is_pad,
                                                      device)
            mot_action = torch.cat([state_target, action_target], dim=-1)
            if state_target_is_pad is not None \
                    and action_target_is_pad is not None:
                mot_action_is_pad = state_target_is_pad | action_target_is_pad
            else:
                mot_action_is_pad = (
                    state_target_is_pad if state_target_is_pad is not None else
                    action_target_is_pad)
        else:
            mot_action = state_target
            mot_action_is_pad = state_target_is_pad

        ret = super().forward(
            input_latents=input_latents,
            context=context,
            context_mask=context_mask,
            action=mot_action,
            action_is_pad=mot_action_is_pad,
            image_is_pad=image_is_pad,
            proprio=proprio,
            training_mode=training_mode,
            **kwargs,
        )
        self._rename_state_loss_keys(
            ret, joint_state_action=self.joint_state_action)

        if self.action_decoder is None or action_target is None:
            # Joint state-action prediction supervises actions inside the
            # MoT slot, so there is no separate decoder loss to add.
            ret['loss_state_to_action'] = ret['loss'].detach().new_zeros(())
            return ret
        if proprio is None:
            raise ValueError('`proprio` is required for state_to_action loss.')
        current_state = proprio[:, 0, :] if proprio.ndim == 3 else proprio
        current_state = self._to_model_dtype(current_state, device)
        action_target = self._to_model_dtype(action_target, device)
        action_target_is_pad = self._to_bool_mask(action_target_is_pad, device)

        loss_state_to_action = self._compute_state_to_action_loss(
            current_state=current_state,
            state_chunks=state_target,
            target_action=action_target,
            state_chunk_is_pad=state_target_is_pad,
            target_action_is_pad=action_target_is_pad,
            embodiment_ids=embodiment_ids,
        )
        loss_state_to_action = (
            self.loss_lambda_state_to_action * loss_state_to_action)
        ret['loss'] = ret['loss'] + loss_state_to_action
        ret['loss_state_to_action'] = loss_state_to_action.detach()
        return ret

    @torch.no_grad()
    def predict_state_chunk(self, *args, **kwargs) -> torch.Tensor:
        pred = super().predict_action(*args, **kwargs)
        if self.joint_state_action:
            return pred[..., :self.state_chunk_dim]
        return pred

    @torch.no_grad()
    def predict_action(
        self,
        *args,
        proprio: Optional[torch.Tensor] = None,
        embodiment_ids: Optional[torch.Tensor] = None,
        return_state_chunks: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if self.joint_state_action:
            pred_joint = super().predict_action(*args, **kwargs)
            pred_state_chunks = pred_joint[..., :self.state_chunk_dim]
            pred_actions = pred_joint[..., self.state_chunk_dim:]
            if return_state_chunks:
                return pred_actions, pred_state_chunks
            return pred_actions
        pred_state_chunks = self.predict_state_chunk(*args, **kwargs)
        if self.action_decoder is None:
            return pred_state_chunks
        if proprio is None:
            raise ValueError('`proprio` is required when using '
                             '`action_decoder` for predict_action.')
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        if proprio.ndim == 3:
            proprio = proprio[:, 0, :]
        device = pred_state_chunks.device
        pred_state_chunks = pred_state_chunks.to(
            device=device, dtype=self.torch_dtype)
        proprio = proprio.to(device=device, dtype=self.torch_dtype)
        pred_actions = self.action_decoder(
            current_state=proprio,
            state_chunks=pred_state_chunks,
            embodiment_ids=embodiment_ids,
        )
        if return_state_chunks:
            return pred_actions, pred_state_chunks
        return pred_actions
