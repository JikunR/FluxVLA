from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from .wan_video_dit import flash_attention, modulate, rope_apply


class MoT(nn.Module):
    def __init__(
        self,
        mixtures: Dict[str, nn.Module],
        mot_checkpoint_mixed_attn: bool = True,
    ):
        super().__init__()
        if not mixtures:
            raise ValueError('`mixtures` cannot be empty.')

        self.mixtures = nn.ModuleDict(mixtures)
        self.expert_order = list(self.mixtures.keys())
        self.mot_checkpoint_mixed_attn = bool(mot_checkpoint_mixed_attn)

        first_expert = self.mixtures[self.expert_order[0]]
        self.num_layers = len(first_expert.blocks)
        self.num_heads = first_expert.num_heads
        self.attn_head_dim = first_expert.attn_head_dim

        for name in self.expert_order[1:]:
            expert = self.mixtures[name]
            if len(expert.blocks) != self.num_layers:
                raise ValueError(
                    f'All experts must have same number of layers; got {self.num_layers} and {len(expert.blocks)}'
                )
            if expert.num_heads != self.num_heads:
                raise ValueError(
                    f'All experts must have same num_heads; got {self.num_heads} and {expert.num_heads}'
                )
            if expert.attn_head_dim != self.attn_head_dim:
                raise ValueError(
                    'All experts must have same attn_head_dim; '
                    f'got {self.attn_head_dim} and {expert.attn_head_dim}'
                )

    @staticmethod
    def _split_modulation(block, t_mod: torch.Tensor):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1

        base_mod = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (base_mod + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            # means t_mod has separate modulation for each token, otherwise same modulation for all tokens in the block
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def _get_expert(self, expert_name: str) -> nn.Module:
        if expert_name not in self.mixtures:
            raise ValueError(
                f'Unknown expert {expert_name!r}; available experts: {self.expert_order}.')
        return self.mixtures[expert_name]

    @staticmethod
    def _validate_attention_mask(
        attention_mask: torch.Tensor,
        expected_seq_len: int,
        mask_name: str = 'attention_mask',
        batch_size: Optional[int] = None,
    ) -> None:
        if attention_mask.ndim == 2:
            if attention_mask.shape[0] != attention_mask.shape[1]:
                raise ValueError(
                    f'`{mask_name}` must be square, got shape {tuple(attention_mask.shape)}')
            if attention_mask.shape[0] != expected_seq_len:
                raise ValueError(
                    f'`{mask_name}` seq length mismatch: '
                    f'mask={attention_mask.shape[0]} vs tokens={expected_seq_len}')
            return
        if attention_mask.ndim == 4:
            if attention_mask.shape[1] != 1:
                raise ValueError(
                    f'`{mask_name}` 4D mask must be [B,1,S,S], got shape '
                    f'{tuple(attention_mask.shape)}')
            if batch_size is not None and attention_mask.shape[0] != batch_size:
                raise ValueError(
                    f'`{mask_name}` batch mismatch: '
                    f'mask={attention_mask.shape[0]} vs tokens={batch_size}')
            if attention_mask.shape[-2:] != (expected_seq_len, expected_seq_len):
                raise ValueError(
                    f'`{mask_name}` seq length mismatch: '
                    f'mask={tuple(attention_mask.shape[-2:])} vs '
                    f'tokens={expected_seq_len}')
            return
        raise ValueError(
            f'`{mask_name}` must be 2D [S,S] or 4D [B,1,S,S], got shape '
            f'{tuple(attention_mask.shape)}')

    def _mixed_attention(
        self,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_mask = attention_mask.to(device=q_cat.device)

        def _forward(q: torch.Tensor, k: torch.Tensor,
                     v: torch.Tensor) -> torch.Tensor:
            return flash_attention(
                q=q,
                k=k,
                v=v,
                num_heads=self.num_heads,
                ctx_mask=attn_mask,
            )

        if self.mot_checkpoint_mixed_attn and self.training:
            return torch.utils.checkpoint.checkpoint(
                _forward,
                q_cat,
                k_cat,
                v_cat,
                use_reentrant=False,
            )
        return _forward(q_cat, k_cat, v_cat)

    @staticmethod
    def _apply_expert_post_block(
        block,
        residual_x: torch.Tensor,
        mixed_attn_out: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        x = block.gate(residual_x, gate_msa, block.self_attn.o(mixed_attn_out))

        if context_payload is not None:
            context = context_payload.get('context')
            if context is not None:
                context_mask = context_payload.get('mask')
                if context_mask is not None and context_mask.dim() == 3:
                    context_mask = context_mask.unsqueeze(1)
                x = x + block.cross_attn(block.norm3(x), context, ctx_mask=context_mask)

        mlp_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
        x = block.gate(x, gate_mlp, block.ffn(mlp_input))
        return x

    def _build_expert_attention_io(
        self,
        expert,
        block,
        x: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        bool,
    ]:
        """Build per-expert attention tensors and post-block states.

        Args:
            expert: Expert module that owns this `block`; only used to read
                `use_gradient_checkpointing`.
            block: Transformer block for current layer (`expert.blocks[layer_idx]`).
            x: Current expert tokens, shape [B, S, D].
            freqs: RoPE frequencies aligned with token sequence, shape [S, 1, rope_dim].
            t_mod: Time modulation tensor for this expert/layer.

        Returns:
            q: Query after q-proj, RMSNorm, and RoPE, shape [B, S, H*Dh].
            k: Key after k-proj, RMSNorm, and RoPE, shape [B, S, H*Dh].
            v: Value after v-proj, shape [B, S, H*Dh].
            residual_x: Original input `x` for residual path in post block.
            gate_msa: Gating tensor for self-attention residual branch.
            shift_mlp: Shift tensor for MLP modulation.
            scale_mlp: Scale tensor for MLP modulation.
            gate_mlp: Gating tensor for MLP residual branch.
            use_gradient_checkpointing: Whether this expert enables
                checkpointing.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._split_modulation(block, t_mod)
        attn_input = modulate(block.norm1(x), shift_msa, scale_msa)

        q = block.self_attn.norm_q(block.self_attn.q(attn_input))
        k = block.self_attn.norm_k(block.self_attn.k(attn_input))
        v = block.self_attn.v(attn_input)

        q = rope_apply(q, freqs, block.num_heads)
        k = rope_apply(k, freqs, block.num_heads)

        use_gradient_checkpointing = bool(
            getattr(expert, 'use_gradient_checkpointing', False))
        return (
            q,
            k,
            v,
            x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            use_gradient_checkpointing,
        )

    def _apply_post_block(
        self,
        block,
        residual_x: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        use_gradient_checkpointing: bool,
        mixed_slice: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        """Apply post-attention computations for one expert block.

        Args:
            block: Transformer block for current layer.
            residual_x: Residual input tokens before attention update, shape [B, S, D].
            gate_msa: Gating tensor used after mixed self-attention.
            shift_mlp: Shift tensor for MLP input modulation.
            scale_mlp: Scale tensor for MLP input modulation.
            gate_mlp: Gating tensor used after MLP.
            use_gradient_checkpointing: If True and training, checkpoint this
                post block.
            mixed_slice: Mixed-attention output for this expert, shape [B, S, H*Dh].
            context_payload: Optional dict for cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, S, L] or [B, 1, S, L]

        Returns:
            Updated expert tokens after self-attn residual, optional cross-attn, and MLP.
        """
        def _post_fn(
            _mixed_slice: torch.Tensor,
            _residual_x: torch.Tensor,
            _gate_msa: torch.Tensor,
            _shift_mlp: torch.Tensor,
            _scale_mlp: torch.Tensor,
            _gate_mlp: torch.Tensor,
            _context_payload: Optional[dict],
        ) -> torch.Tensor:
            return self._apply_expert_post_block(
                block=block,
                residual_x=_residual_x,
                mixed_attn_out=_mixed_slice,
                gate_msa=_gate_msa,
                shift_mlp=_shift_mlp,
                scale_mlp=_scale_mlp,
                gate_mlp=_gate_mlp,
                context_payload=_context_payload,
            )

        if use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                _post_fn,
                mixed_slice,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                context_payload,
                use_reentrant=False,
            )
        return _post_fn(
            mixed_slice,
            residual_x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            context_payload,
        )

    def prefill_expert_cache(
        self,
        expert_name: str,
        tokens: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
        context_payload: Optional[dict],
        attention_mask: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        """Run one expert once and cache its per-layer K/V for later mixed attention.

        Args:
            expert_name: Expert to prefill.
            tokens: Expert tokens before layer 0, shape [B, S, D].
            freqs: RoPE frequencies aligned with tokens, shape [S, 1, rope_dim].
            t_mod: Time modulation tensor for this expert.
            context_payload: Optional dict for cross-attention.
                - `context`: encoder states [B, L, D].
                - `mask`: attention mask [B, S, L] or [B, 1, S, L].
            attention_mask: Expert self-attention mask, shape [S, S].

        Returns:
            Layer-wise cache list with length `num_layers`.
            Each entry contains:
                - `k`: key tensor [B, S, H*Dh].
                - `v`: value tensor [B, S, H*Dh].
        """
        expert = self._get_expert(expert_name)
        self._validate_attention_mask(
            attention_mask,
            expected_seq_len=int(tokens.shape[1]),
        )

        x = tokens
        kv_cache: list[dict[str, torch.Tensor]] = []
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            (
                q,
                k,
                v,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=freqs,
                t_mod=t_mod,
            )
            mixed = self._mixed_attention(
                q_cat=q,
                k_cat=k,
                v_cat=v,
                attention_mask=attention_mask,
            )
            x = self._apply_post_block(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=context_payload,
            )
            kv_cache.append({'k': k, 'v': v})
        return kv_cache

    def forward_expert_with_cache(
        self,
        expert_name: str,
        tokens: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
        context_payload: Optional[dict],
        kv_cache_by_expert: Dict[str, list[dict[str, torch.Tensor]]],
        attention_mask: torch.Tensor,
        attention_order: Sequence[str],
    ) -> torch.Tensor:
        """Run one expert while attending to cached K/V from other experts.

        Args:
            expert_name: Expert to update from its current tokens.
            tokens: Current expert tokens before layer 0, shape [B, S, D].
            freqs: RoPE frequencies aligned with `tokens`, shape [S, 1, rope_dim].
            t_mod: Time modulation tensor for `expert_name`.
            context_payload: Optional dict for this expert's cross-attention.
                - `context`: encoder states [B, L, D].
                - `mask`: attention mask [B, S, L] or [B, 1, S, L].
            kv_cache_by_expert: Cached K/V for every non-current expert named
                in `attention_order`.
            attention_mask: Joint attention mask in `attention_order`, shape
                [sum(S_i), sum(S_i)].
            attention_order: Expert order used to concatenate cached/current K/V
                and slice query rows from `attention_mask`. Must contain
                `expert_name` exactly once.

        Returns:
            Updated tokens for `expert_name` after all layers, shape [B, S, D].
        """
        expert = self._get_expert(expert_name)
        attention_order = list(attention_order)
        if attention_order.count(expert_name) != 1:
            raise ValueError(
                f'`attention_order` must contain {expert_name!r} exactly once, '
                f'got {attention_order}.')
        if len(set(attention_order)) != len(attention_order):
            raise ValueError(f'`attention_order` contains duplicates: {attention_order}.')

        unknown = [name for name in attention_order if name not in self.mixtures]
        if unknown:
            raise ValueError(f'Unknown experts in `attention_order`: {unknown}.')

        current_seq_len = int(tokens.shape[1])
        seq_lens: dict[str, int] = {expert_name: current_seq_len}
        for cached_name in attention_order:
            if cached_name == expert_name:
                continue
            if cached_name not in kv_cache_by_expert:
                raise ValueError(f'Missing K/V cache for expert {cached_name!r}.')
            cached_layers = kv_cache_by_expert[cached_name]
            if len(cached_layers) != self.num_layers:
                raise ValueError(
                    f'K/V cache for expert {cached_name!r} must contain '
                    f'{self.num_layers} layers, got {len(cached_layers)}.')
            first_cache = cached_layers[0]
            if 'k' not in first_cache or 'v' not in first_cache:
                raise ValueError(
                    f'K/V cache for expert {cached_name!r} layer 0 must contain `k` and `v`.')
            if first_cache['k'].shape[1] != first_cache['v'].shape[1]:
                raise ValueError(
                    f'K/V cache for expert {cached_name!r} layer 0 has mismatched seq lens.')
            seq_lens[cached_name] = int(first_cache['k'].shape[1])

        total_seq_len = sum(seq_lens[name] for name in attention_order)
        self._validate_attention_mask(attention_mask, expected_seq_len=total_seq_len)

        query_start = 0
        for name in attention_order:
            if name == expert_name:
                break
            query_start += seq_lens[name]
        query_end = query_start + current_seq_len
        if attention_mask.ndim == 4:
            query_attention_mask = attention_mask[:, :, query_start:query_end,
                                                  :total_seq_len]
        else:
            query_attention_mask = attention_mask[query_start:query_end,
                                                  :total_seq_len]

        x = tokens
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            (
                q_current,
                k_current,
                v_current,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=freqs,
                t_mod=t_mod,
            )

            k_chunks = []
            v_chunks = []
            for name in attention_order:
                if name == expert_name:
                    k_chunks.append(k_current)
                    v_chunks.append(v_current)
                    continue
                layer_cache = kv_cache_by_expert[name][layer_idx]
                if 'k' not in layer_cache or 'v' not in layer_cache:
                    raise ValueError(
                        f'K/V cache for expert {name!r} layer {layer_idx} '
                        'must contain `k` and `v`.')
                k_cached = layer_cache['k']
                v_cached = layer_cache['v']
                if k_cached.shape[1] != seq_lens[name] or v_cached.shape[1] != seq_lens[name]:
                    raise ValueError(
                        f'K/V cache for expert {name!r} layer {layer_idx} '
                        f'seq len mismatch, expected {seq_lens[name]}.')
                k_chunks.append(k_cached)
                v_chunks.append(v_cached)

            k_cat = torch.cat(k_chunks, dim=1)
            v_cat = torch.cat(v_chunks, dim=1)
            mixed = self._mixed_attention(
                q_cat=q_current,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=query_attention_mask,
            )
            x = self._apply_post_block(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=context_payload,
            )
        return x

    def forward(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
    ):
        missing = [k for k in self.expert_order if k not in embeds_all]
        if missing:
            raise ValueError(f'Missing expert tokens for {missing}')
        missing = [k for k in self.expert_order if k not in freqs_all]
        if missing:
            raise ValueError(f'Missing expert freqs for {missing}')
        missing = [k for k in self.expert_order if k not in t_mod_all]
        if missing:
            raise ValueError(f'Missing expert t_mod for {missing}')

        tokens_all = {k: v for k, v in embeds_all.items()}

        for layer_idx in range(self.num_layers):
            q_chunks = []
            k_chunks = []
            v_chunks = []
            cached = {}
            seq_lens = []

            for name in self.expert_order:
                expert = self.mixtures[name]
                block = expert.blocks[layer_idx]
                x = tokens_all[name]
                freqs = freqs_all[name]
                t_mod = t_mod_all[name]

                (
                    q,
                    k,
                    v,
                    residual_x,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                    use_gradient_checkpointing,
                ) = self._build_expert_attention_io(
                    expert=expert,
                    block=block,
                    x=x,
                    freqs=freqs,
                    t_mod=t_mod,
                )

                q_chunks.append(q)
                k_chunks.append(k)
                v_chunks.append(v)
                seq_lens.append(x.shape[1])
                cached[name] = {
                    'block': block,
                    'residual_x': residual_x,
                    'gate_msa': gate_msa,
                    'shift_mlp': shift_mlp,
                    'scale_mlp': scale_mlp,
                    'gate_mlp': gate_mlp,
                    'use_gradient_checkpointing': use_gradient_checkpointing,
                }

            # 3. concat all tokens for mixed attention
            q_cat = torch.cat(q_chunks, dim=1)
            k_cat = torch.cat(k_chunks, dim=1)
            v_cat = torch.cat(v_chunks, dim=1)

            total_seq = q_cat.shape[1]
            self._validate_attention_mask(
                attention_mask,
                expected_seq_len=int(total_seq),
                batch_size=int(q_cat.shape[0]),
            )

            mixed = self._mixed_attention(q_cat=q_cat, k_cat=k_cat, v_cat=v_cat, attention_mask=attention_mask)

            start = 0
            for name, seq_len in zip(self.expert_order, seq_lens):
                # 4. split mixed attention output and apply post-attention blocks for each expert
                end = start + seq_len
                mixed_slice = mixed[:, start:end, :]
                cached_expert = cached[name]
                block = cached_expert['block']
                context_payload = context_all.get(name)

                updated_tokens = self._apply_post_block(
                    block=block,
                    residual_x=cached_expert['residual_x'],
                    gate_msa=cached_expert['gate_msa'],
                    shift_mlp=cached_expert['shift_mlp'],
                    scale_mlp=cached_expert['scale_mlp'],
                    gate_mlp=cached_expert['gate_mlp'],
                    use_gradient_checkpointing=cached_expert[
                        'use_gradient_checkpointing'],
                    mixed_slice=mixed_slice,
                    context_payload=context_payload,
                )

                tokens_all[name] = updated_tokens
                start = end

        return tokens_all
