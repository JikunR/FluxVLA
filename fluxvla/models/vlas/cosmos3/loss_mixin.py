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


class Cosmos3LossMixin:
    """Flow-matching losses for action and Wan latent predictions."""

    def _compute_action_loss(
        self,
        preds_action: List[torch.Tensor],
        target_action: Optional[List[torch.Tensor]],
        raw_action_dim: Optional[List[torch.Tensor]],
        condition_mask: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        if target_action is None:
            if preds_action:
                return 0.0 * sum(pred.sum() for pred in preds_action)
            return next(self.parameters()).sum() * 0.0
        if not preds_action:
            return next(self.parameters()).sum() * 0.0

        normalize_by_active = bool(
            self.rectified_flow_training_config['normalize_loss_by_active'])
        losses = []
        for index, (pred,
                    target) in enumerate(zip(preds_action, target_action)):
            valid_dim = int(raw_action_dim[index].item()
                            ) if raw_action_dim is not None else pred.shape[-1]
            sqerr = (pred[:, :valid_dim].float() -
                     target[:, :valid_dim].float())**2
            if condition_mask is None:
                losses.append(sqerr.mean())
                continue
            noisy_mask = 1.0 - condition_mask[index].to(
                device=sqerr.device, dtype=sqerr.dtype)
            if normalize_by_active:
                active_count = (noisy_mask.sum() *
                                (sqerr.numel() // noisy_mask.numel())).clamp(
                                    min=1)
                losses.append((sqerr * noisy_mask).sum() / active_count)
            else:
                losses.append((sqerr * noisy_mask).mean())
        return torch.stack(losses).mean() if losses else next(
            self.parameters()).sum() * 0.0

    def _compute_vision_loss(
        self,
        preds_vision: List[torch.Tensor],
        target_vision: Optional[List[torch.Tensor]],
        condition_mask: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        if target_vision is None:
            if preds_vision:
                return 0.0 * sum(pred.sum() for pred in preds_vision)
            return next(self.parameters()).sum() * 0.0
        if not preds_vision:
            return next(self.parameters()).sum() * 0.0
        normalize_by_active = bool(
            self.rectified_flow_training_config['normalize_loss_by_active'])
        losses = []
        for index, (pred,
                    target) in enumerate(zip(preds_vision, target_vision)):
            sqerr = (pred.float() - target.float())**2
            if condition_mask is None:
                losses.append(sqerr.mean())
                continue
            noisy_mask = 1.0 - condition_mask[index].to(
                device=sqerr.device, dtype=sqerr.dtype)
            noisy_mask = noisy_mask.view(1, 1, noisy_mask.shape[0], 1, 1)
            if normalize_by_active:
                active_count = (noisy_mask.sum() *
                                (sqerr.numel() // noisy_mask.numel())).clamp(
                                    min=1)
                losses.append((sqerr * noisy_mask).sum() / active_count)
            else:
                losses.append((sqerr * noisy_mask).mean())
        return torch.stack(losses).mean() if losses else next(
            self.parameters()).sum() * 0.0
