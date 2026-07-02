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

from typing import Optional

import torch
import torch.nn as nn


class StateToActionDecoder(nn.Module):
    """Shared decoder from state transitions to action chunks.

    Embodiment information is injected as a lightweight prefix token plus an
    optional hidden-state FiLM. The main projections remain shared across
    embodiments, avoiding a separate decoder per controller version.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 1024,
        ffn_dim: Optional[int] = None,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.0,
        max_horizon: int = 1024,
        max_num_embodiments: int = 32,
        use_embodiment_film: bool = True,
        use_delta: bool = True,
        device: str = 'cpu',
        torch_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_horizon = int(max_horizon)
        self.max_num_embodiments = int(max_num_embodiments)
        self.use_embodiment_film = bool(use_embodiment_film)
        self.use_delta = bool(use_delta)
        if self.max_num_embodiments <= 0:
            raise ValueError('`max_num_embodiments` must be positive, got '
                             f'{self.max_num_embodiments}.')

        input_dim = self.state_dim * (3 if self.use_delta else 2)
        self.input_proj = nn.Linear(input_dim, self.hidden_dim)
        self.pos_embedding = nn.Embedding(self.max_horizon, self.hidden_dim)
        self.embodiment_embedding = nn.Embedding(
            self.max_num_embodiments,
            self.hidden_dim,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=int(ffn_dim or self.hidden_dim * 4),
            dropout=float(dropout),
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(num_layers),
        )
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.embodiment_film = (
            nn.Embedding(self.max_num_embodiments, 2 * self.hidden_dim)
            if self.use_embodiment_film else None)
        self.output_proj = nn.Linear(self.hidden_dim, self.action_dim)
        nn.init.normal_(self.embodiment_embedding.weight, std=0.02)
        if self.embodiment_film is not None:
            nn.init.zeros_(self.embodiment_film.weight)
        self.to(device=device, dtype=torch_dtype)

    def _prepare_embodiment_ids(
        self,
        embodiment_ids: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if embodiment_ids is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        embodiment_ids = torch.as_tensor(
            embodiment_ids, dtype=torch.long, device=device)
        if embodiment_ids.ndim == 0:
            embodiment_ids = embodiment_ids.expand(batch_size)
        else:
            embodiment_ids = embodiment_ids.reshape(-1)
        if embodiment_ids.numel() != batch_size:
            raise ValueError('`embodiment_ids` size must match batch size '
                             f'{batch_size}, got {embodiment_ids.numel()}.')
        if embodiment_ids.numel() > 0:
            min_id = int(embodiment_ids.min().item())
            max_id = int(embodiment_ids.max().item())
            if min_id < 0 or max_id >= self.max_num_embodiments:
                raise ValueError('`embodiment_ids` out of range for '
                                 f'max_num_embodiments='
                                 f'{self.max_num_embodiments}: '
                                 f'min={min_id}, max={max_id}.')
        return embodiment_ids

    def forward(
        self,
        current_state: torch.Tensor,
        state_chunks: torch.Tensor,
        state_masks: Optional[torch.Tensor] = None,
        embodiment_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if current_state.ndim == 3:
            current_state = current_state[:, 0]
        if current_state.ndim != 2:
            raise ValueError('`current_state` must be [B,D] or [B,1,D], got '
                             f'{tuple(current_state.shape)}')
        if state_chunks.ndim != 3:
            raise ValueError('`state_chunks` must be [B,T,D], got '
                             f'{tuple(state_chunks.shape)}')
        if current_state.shape[0] != state_chunks.shape[0]:
            raise ValueError('Batch mismatch between current_state and '
                             f'state_chunks: {current_state.shape[0]} vs '
                             f'{state_chunks.shape[0]}')
        if current_state.shape[-1] != self.state_dim:
            raise ValueError(f'`current_state` last dim must be '
                             f'{self.state_dim}, got '
                             f'{current_state.shape[-1]}')
        if state_chunks.shape[-1] != self.state_dim:
            raise ValueError(f'`state_chunks` last dim must be '
                             f'{self.state_dim}, got '
                             f'{state_chunks.shape[-1]}')

        horizon = state_chunks.shape[1]
        if horizon > self.max_horizon:
            raise ValueError(f'State horizon {horizon} exceeds '
                             f'max_horizon={self.max_horizon}.')
        embodiment_ids = self._prepare_embodiment_ids(
            embodiment_ids=embodiment_ids,
            batch_size=int(state_chunks.shape[0]),
            device=state_chunks.device,
        )

        prev_states = torch.cat(
            [current_state.unsqueeze(1), state_chunks[:, :-1]],
            dim=1,
        )
        features = [prev_states, state_chunks]
        if self.use_delta:
            features.append(state_chunks - prev_states)
        tokens = self.input_proj(torch.cat(features, dim=-1))
        positions = torch.arange(
            horizon, device=state_chunks.device, dtype=torch.long)
        tokens = tokens + self.pos_embedding(positions).unsqueeze(0)
        tokens = torch.cat(
            [self.embodiment_embedding(embodiment_ids).unsqueeze(1), tokens],
            dim=1,
        )

        key_padding_mask = None
        if state_masks is not None:
            state_padding_mask = ~state_masks.to(
                device=state_chunks.device, dtype=torch.bool)
            embodiment_padding_mask = torch.zeros(
                (state_chunks.shape[0], 1),
                dtype=torch.bool,
                device=state_chunks.device,
            )
            key_padding_mask = torch.cat(
                [embodiment_padding_mask, state_padding_mask], dim=1)
        tokens = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        tokens = self.norm(tokens[:, 1:])
        if self.embodiment_film is not None:
            scale, shift = self.embodiment_film(embodiment_ids).chunk(
                2, dim=-1)
            tokens = tokens * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.output_proj(tokens)
