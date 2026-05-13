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
"""Muon optimizer: Momentum + Newton-Schulz Orthogonalization.

Reference: https://arxiv.org/abs/2502.16982
"""

import math

import torch
from torch.optim import Optimizer


def _zeropower_via_newtonschulz5(G: torch.Tensor,
                                 steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration approximating the matrix sign function.

    Computes the orthogonal matrix closest to G, equivalent to UV^T where
    G = USV^T is the SVD decomposition. Uses quintic iteration coefficients
    that maximise the slope at the origin for faster convergence.

    Args:
        G: A 2D tensor (matrix).
        steps: Number of Newton-Schulz iterations (default 5).

    Returns:
        Orthogonalized matrix approximating UV^T.
    """
    assert G.ndim == 2, f'Expected 2D tensor, got {G.ndim}D'

    X = G / (G.norm() + 1e-7)

    if G.size(-2) > G.size(-1):
        X = X.T

    # Quintic Newton-Schulz iteration coefficients
    coeffs = [
        (3.4445, -4.7750, 2.0315),
        (2.3863, -1.9655, 0.5792),
        (1.7921, -0.8873, 0.0952),
        (1.4561, -0.4183, -0.0378),
        (1.2391, -0.1690, -0.0700),
    ]

    for i in range(min(steps, len(coeffs))):
        a, b, c = coeffs[i]
        XT = X.T
        XXT = X @ XT
        X = a * X + b * (XXT @ X) + c * (XXT @ XXT @ X)

    if G.size(-2) > G.size(-1):
        X = X.T

    return X


class Muon(Optimizer):
    """Muon optimizer: Momentum + Newton-Schulz Orthogonalization.

    For matrix parameters (ndim >= 2), applies momentum followed by
    Newton-Schulz orthogonalization of the update. For non-matrix
    parameters (1D biases, layer-norm scales, embeddings), falls back
    to AdamW.

    Reference: https://arxiv.org/abs/2502.16982

    Args:
        params: Iterable of parameters or parameter groups.
        lr: Learning rate (default 1e-3).
        mu: Momentum coefficient (default 0.95).
        betas: AdamW (beta1, beta2) for non-matrix parameters
            (default (0.9, 0.95)).
        weight_decay: Weight decay coefficient (default 0.1).
        nesterov: Whether to use Nesterov momentum (default False).
        ns_steps: Newton-Schulz iteration steps (default 5).
        adamw_eps: Epsilon for AdamW fallback (default 1e-8).
        adjust_lr: Per-parameter LR scaling. ``"rms_norm"`` scales lr by
            rms(W) / max(rms(update), 1e-7); ``None`` disables it.
            Default ``"rms_norm"``.
        matrix_lr_scale: Factor applied to lr for matrix parameters.
            When set to None (default), uses the same lr for both groups.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        mu: float = 0.95,
        betas: tuple = (0.9, 0.95),
        weight_decay: float = 0.1,
        nesterov: bool = False,
        ns_steps: int = 5,
        adamw_eps: float = 1e-8,
        adjust_lr: str | None = 'rms_norm',
        matrix_lr_scale: float | None = None,
    ):
        if not 0.0 <= lr:
            raise ValueError(f'Invalid learning rate: {lr}')
        if not 0.0 <= mu < 1.0:
            raise ValueError(f'Invalid mu parameter: {mu}')
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f'Invalid beta1 parameter: {betas[0]}')
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f'Invalid beta2 parameter: {betas[1]}')
        if not 0.0 <= weight_decay:
            raise ValueError(f'Invalid weight_decay: {weight_decay}')

        defaults = dict(
            lr=lr,
            mu=mu,
            betas=betas,
            weight_decay=weight_decay,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_eps=adamw_eps,
            adjust_lr=adjust_lr,
            matrix_lr_scale=matrix_lr_scale,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _rms(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.norm(2) / (tensor.numel()**0.5)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            mu = group['mu']
            beta1, beta2 = group['betas']
            weight_decay = group['weight_decay']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            adamw_eps = group['adamw_eps']
            adjust_lr = group['adjust_lr']
            matrix_lr_scale = group.get('matrix_lr_scale', None)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        'Muon does not support sparse gradients')

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    if p.ndim >= 2:
                        state['momentum'] = torch.zeros_like(p)
                    else:
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1

                if p.ndim >= 2:
                    # ---- Muon update for matrix parameters ----
                    momentum = state['momentum']

                    if weight_decay != 0:
                        grad = grad + weight_decay * p.data

                    momentum.mul_(mu).add_(grad)

                    if nesterov:
                        update = momentum.clone().mul_(mu).add_(grad)
                    else:
                        update = momentum.clone()

                    update = _zeropower_via_newtonschulz5(
                        update, steps=ns_steps)

                    if adjust_lr == 'rms_norm':
                        param_rms = self._rms(p.data)
                        update_rms = self._rms(update)
                        scale = param_rms / max(update_rms, 1e-7)
                        scaled_lr = lr * max(scale, 1.0)
                    else:
                        scaled_lr = lr

                    if matrix_lr_scale is not None:
                        scaled_lr = scaled_lr * matrix_lr_scale

                    p.data.add_(update, alpha=-scaled_lr)

                else:
                    # ---- AdamW update for non-matrix parameters ----
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    step = state['step']

                    bias_correction1 = 1 - beta1**step
                    bias_correction2 = 1 - beta2**step

                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(
                        grad, grad, value=1 - beta2)

                    denom = (exp_avg_sq.sqrt() /
                             math.sqrt(bias_correction2)).add_(adamw_eps)
                    step_size = lr / bias_correction1

                    if weight_decay != 0:
                        p.data.mul_(1 - lr * weight_decay)

                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
