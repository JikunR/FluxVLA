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

from typing import Any, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_2tuple(value: Sequence[int] | int, name: str) -> Tuple[int, int]:
    if isinstance(value, int):
        return int(value), int(value)
    if len(value) != 2:
        raise ValueError(f'`{name}` must contain 2 integers, got {value}.')
    return int(value[0]), int(value[1])


class DirectDINOFeatureHead(nn.Module):
    """Project MoT video hidden tokens directly to future DINO features."""

    def __init__(
        self,
        hidden_dim: int,
        target_dim: int,
        target_grid_size: Optional[Sequence[int] | int] = None,
        device: str = 'cpu',
        torch_dtype: torch.dtype = torch.float32,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.target_dim = int(target_dim)
        self.target_grid_size = (None
                                 if target_grid_size is None else _as_2tuple(
                                     target_grid_size, 'target_grid_size'))
        self.proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.target_dim),
        ).to(
            device=device, dtype=torch_dtype)

    def forward(
        self,
        video_tokens: torch.Tensor,
        video_pre: dict,
    ) -> torch.Tensor:
        grid_size = video_pre['meta']['grid_size']
        bsz = video_tokens.shape[0]
        steps, height, width = map(int, grid_size)
        tokens = video_tokens.reshape(bsz, steps, height, width,
                                      self.hidden_dim)
        pred = self.proj(tokens).reshape(bsz, steps, height, width,
                                         self.target_dim)

        target_grid = self.target_grid_size
        if target_grid is not None and target_grid != (height, width):
            pred_2d = pred.permute(0, 1, 4, 2,
                                   3).reshape(bsz * steps, self.target_dim,
                                              height, width)
            pred_2d = F.interpolate(
                pred_2d,
                size=target_grid,
                mode='bilinear',
                align_corners=False,
            )
            tgt_h, tgt_w = target_grid
            pred = pred_2d.reshape(bsz, steps, self.target_dim, tgt_h,
                                   tgt_w).permute(0, 1, 3, 4, 2)
        return pred.reshape(bsz, steps, -1, self.target_dim)


class _SemanticResamplerBlock(nn.Module):
    """Perceiver-style resampler block for semantic patch decoding."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_mult: int,
        dropout: float,
        device: str,
        torch_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.source_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        ffn_dim = int(dim) * int(ff_mult)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )
        self.to(device=device, dtype=torch_dtype)

    def forward(self, target_tokens: torch.Tensor,
                source_tokens: torch.Tensor) -> torch.Tensor:
        source_tokens = self.source_norm(source_tokens)
        target_query = self.query_norm(target_tokens)
        memory = torch.cat([source_tokens, target_query], dim=1)
        attn_out, _ = self.cross_attn(
            query=target_query,
            key=memory,
            value=memory,
            need_weights=False,
        )
        target_tokens = target_tokens + attn_out
        return target_tokens + self.ffn(target_tokens)


class SemanticQueryAdapter(nn.Module):
    """Video-side semantic query adapter decoded to semantic features.

    This adapter does not run its own attention. ``pre_dit`` creates semantic
    query tokens that are appended to the video mixture, so the existing video
    expert blocks own Q/K/V, norms, FFN, and cross-modal interaction inside
    MoT.
    """

    uses_video_query_tokens = True

    def __init__(
        self,
        hidden_dim: int,
        target_dim: int,
        target_grid_size: Optional[Sequence[int] | int] = None,
        query_grid_size: Optional[Sequence[int] | int] = None,
        max_temporal_steps: int = 16,
        max_height: int = 64,
        max_width: int = 64,
        query_init_std: float = 0.02,
        resampler_num_layers: int = 1,
        resampler_num_heads: int = 8,
        resampler_ff_mult: int = 1,
        resampler_dropout: float = 0.0,
        device: str = 'cpu',
        torch_dtype: torch.dtype = torch.float32,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.target_dim = int(target_dim)
        self.target_grid_size = (None
                                 if target_grid_size is None else _as_2tuple(
                                     target_grid_size, 'target_grid_size'))
        self.query_grid_size = (None if query_grid_size is None else
                                _as_2tuple(query_grid_size, 'query_grid_size'))
        self.max_temporal_steps = int(max_temporal_steps)
        self.max_height = int(max_height)
        self.max_width = int(max_width)
        if self.max_temporal_steps <= 0:
            raise ValueError('`max_temporal_steps` must be positive.')
        if self.max_height <= 0 or self.max_width <= 0:
            raise ValueError('`max_height` and `max_width` must be positive.')
        self.resampler_num_layers = int(resampler_num_layers)
        self.resampler_num_heads = int(resampler_num_heads)
        self.resampler_ff_mult = int(resampler_ff_mult)
        if self.resampler_num_layers <= 0:
            raise ValueError('`resampler_num_layers` must be positive.')
        if self.resampler_num_heads <= 0:
            raise ValueError('`resampler_num_heads` must be positive.')
        if self.resampler_ff_mult <= 0:
            raise ValueError('`resampler_ff_mult` must be positive.')
        if self.target_dim % self.resampler_num_heads != 0:
            raise ValueError(
                '`target_dim` must be divisible by `resampler_num_heads`, got '
                f'{self.target_dim} and {self.resampler_num_heads}.')

        init_std = float(query_init_std)
        self.base_query = nn.Parameter(
            torch.randn(1, 1, 1, 1, self.hidden_dim) * init_std)
        self.type_embed = nn.Parameter(
            torch.randn(1, 1, 1, 1, self.hidden_dim) * init_std)
        self.temporal_embed = nn.Parameter(
            torch.randn(1, self.max_temporal_steps, 1, 1, self.hidden_dim) *
            init_std)
        self.height_embed = nn.Parameter(
            torch.randn(1, 1, self.max_height, 1, self.hidden_dim) * init_std)
        self.width_embed = nn.Parameter(
            torch.randn(1, 1, 1, self.max_width, self.hidden_dim) * init_std)
        self.source_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.target_dim),
        ).to(
            device=device, dtype=torch_dtype)
        self.target_base_query = nn.Parameter(
            torch.randn(1, 1, 1, 1, self.target_dim) * init_std)
        self.target_temporal_embed = nn.Parameter(
            torch.randn(1, self.max_temporal_steps, 1, 1, self.target_dim) *
            init_std)
        self.target_height_embed = nn.Parameter(
            torch.randn(1, 1, self.max_height, 1, self.target_dim) * init_std)
        self.target_width_embed = nn.Parameter(
            torch.randn(1, 1, 1, self.max_width, self.target_dim) * init_std)
        self.resampler = nn.ModuleList([
            _SemanticResamplerBlock(
                dim=self.target_dim,
                num_heads=self.resampler_num_heads,
                ff_mult=self.resampler_ff_mult,
                dropout=float(resampler_dropout),
                device=device,
                torch_dtype=torch_dtype,
            ) for _ in range(self.resampler_num_layers)
        ])
        self.to(device=device, dtype=torch_dtype)

    @staticmethod
    def _nearest_video_token_indices(
        steps: int,
        video_height: int,
        video_width: int,
        query_height: int,
        query_width: int,
        device: torch.device,
    ) -> torch.Tensor:
        step_indices = torch.arange(steps, device=device, dtype=torch.long)
        height_indices = torch.linspace(
            0,
            video_height - 1,
            query_height,
            device=device,
        ).round().to(dtype=torch.long)
        width_indices = torch.linspace(
            0,
            video_width - 1,
            query_width,
            device=device,
        ).round().to(dtype=torch.long)
        step_offsets = step_indices.view(steps, 1,
                                         1) * video_height * video_width
        grid_indices = (
            height_indices.view(1, query_height, 1) * video_width +
            width_indices.view(1, 1, query_width))
        return (step_offsets + grid_indices).reshape(-1)

    def pre_dit(
        self,
        video_pre: dict,
    ) -> dict:
        video_tokens = video_pre['tokens']
        batch_size = int(video_tokens.shape[0])
        steps, video_height, video_width = map(int,
                                               video_pre['meta']['grid_size'])
        query_height, query_width = self.query_grid_size or (video_height,
                                                             video_width)
        if steps > self.max_temporal_steps:
            raise ValueError(f'Semantic query steps={steps} exceeds '
                             f'max_temporal_steps={self.max_temporal_steps}.')
        if query_height > self.max_height or query_width > self.max_width:
            raise ValueError(
                f'Semantic query grid {(query_height, query_width)} exceeds '
                f'max grid {(self.max_height, self.max_width)}.')

        query = (
            self.base_query + self.type_embed +
            self.temporal_embed[:, :steps] +
            self.height_embed[:, :, :query_height] +
            self.width_embed[:, :, :, :query_width])
        tokens = query.reshape(1, steps * query_height * query_width,
                               self.hidden_dim).expand(batch_size, -1, -1)
        tokens = tokens.to(
            device=video_tokens.device,
            dtype=video_tokens.dtype,
        ).contiguous()
        source_video_indices = self._nearest_video_token_indices(
            steps=steps,
            video_height=video_height,
            video_width=video_width,
            query_height=query_height,
            query_width=query_width,
            device=video_tokens.device,
        )

        return {
            'tokens': tokens,
            'source_video_indices': source_video_indices,
            'meta': {
                'grid_size': (steps, query_height, query_width),
                'source_grid_size': (steps, video_height, video_width),
                'tokens_per_frame': query_height * query_width,
                'batch_size': batch_size,
            },
        }

    def post_dit(
        self,
        semantic_tokens: torch.Tensor,
        semantic_pre: dict,
    ) -> torch.Tensor:
        steps, query_height, query_width = map(
            int, semantic_pre['meta']['grid_size'])
        bsz = int(semantic_tokens.shape[0])
        target_height, target_width = self.target_grid_size or (query_height,
                                                                query_width)
        if steps > self.max_temporal_steps:
            raise ValueError(f'Semantic target steps={steps} exceeds '
                             f'max_temporal_steps={self.max_temporal_steps}.')
        if target_height > self.max_height or target_width > self.max_width:
            raise ValueError(
                f'Semantic target grid {(target_height, target_width)} '
                'exceeds '
                f'max grid {(self.max_height, self.max_width)}.')

        source = self.source_proj(semantic_tokens).reshape(
            bsz, steps, query_height * query_width, self.target_dim)
        target = (
            self.target_base_query + self.target_temporal_embed[:, :steps] +
            self.target_height_embed[:, :, :target_height] +
            self.target_width_embed[:, :, :, :target_width])
        target = target.reshape(1, steps, target_height * target_width,
                                self.target_dim).expand(bsz, -1, -1, -1)

        source = source.reshape(bsz * steps, query_height * query_width,
                                self.target_dim)
        pred = target.reshape(bsz * steps, target_height * target_width,
                              self.target_dim)
        for block in self.resampler:
            pred = block(pred, source)
        pred = pred.reshape(bsz, steps, target_height * target_width,
                            self.target_dim)
        return pred.reshape(bsz, steps, -1, self.target_dim)
