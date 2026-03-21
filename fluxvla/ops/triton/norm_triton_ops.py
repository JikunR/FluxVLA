# Origin: Modified from
# Upstream-URL: https://github.com/dexmal/realtime-vla/blob/main/pi0_infer.py
# Upstream-Ref: main
# SPDX-License-Identifier: MIT
# Notes: Attribution normalized; no functional change.

import triton
import triton.language as tl


@triton.jit
def layer_norm_small_kernel(x_ptr,
                            out_ptr,
                            norm_w_ptr,
                            norm_b_ptr,
                            seq_len: tl.constexpr,
                            features: tl.constexpr,
                            eps: tl.constexpr = 1e-5):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)

    MAX_LEN: tl.constexpr = 2048

    for i in range(pid, seq_len, psize):
        x = tl.load(
            x_ptr + i * features + tl.arange(0, MAX_LEN),
            mask=tl.arange(0, MAX_LEN) < features,
            other=0.0)
        mean = tl.sum(x) * (1.0 / features)
        x_centered = x - mean
        var = tl.sum(x_centered * x_centered *
                     (tl.arange(0, MAX_LEN) < features)) * (1.0 / features)
        inv_std = 1.0 / tl.sqrt(var + eps)
        x = x_centered * inv_std
        x = x * tl.load(
            norm_w_ptr + tl.arange(0, MAX_LEN),
            mask=tl.arange(0, MAX_LEN) < features,
            other=0.0)
        x = x + tl.load(
            norm_b_ptr + tl.arange(0, MAX_LEN),
            mask=tl.arange(0, MAX_LEN) < features,
            other=0.0)
        tl.store(
            out_ptr + i * features + tl.arange(0, MAX_LEN),
            x.to(tl.bfloat16),
            mask=tl.arange(0, MAX_LEN) < features)


@triton.jit
def rms_norm_kernel(inp_ptr, out_ptr, seq_len: tl.constexpr,
                    features: tl.constexpr):
    pid = tl.program_id(axis=0)
    psize = tl.num_programs(axis=0)
    BLOCK_SIZE: tl.constexpr = 512
    for i in range(pid, seq_len, psize):
        sum_x = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
        for j in range(0, features, BLOCK_SIZE):
            x = tl.load(inp_ptr + i * features + j + tl.arange(0, BLOCK_SIZE))
            sum_x += x * x
        factor = tl.rsqrt(tl.sum(sum_x) / features + 1e-6)
        for j in range(0, features, BLOCK_SIZE):
            x = tl.load(inp_ptr + i * features + j + tl.arange(0, BLOCK_SIZE))
            x = x * factor
            tl.store(out_ptr + i * features + j + tl.arange(0, BLOCK_SIZE), x)


@triton.jit
def rmsnorm_factor_kernel(inp_ptr,
                          factor_ptr,
                          rows: tl.constexpr,
                          features: tl.constexpr,
                          eps: tl.constexpr = 1e-6,
                          BLOCK_SIZE: tl.constexpr = 1024):
    pid = tl.program_id(axis=0)
    psize = tl.num_programs(axis=0)
    for i in range(pid, rows, psize):
        sum_x = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for j in range(0, features, BLOCK_SIZE):
            offs = j + tl.arange(0, BLOCK_SIZE)
            x = tl.load(
                inp_ptr + i * features + offs, mask=offs < features, other=0.0)
            sum_x += x * x
        factor = tl.rsqrt(tl.sum(sum_x) / features + eps)
        tl.store(factor_ptr + i, factor)


@triton.jit
def adarms_norm_kernel(x_ptr, style_ptr, normed_x_ptr, gate_ptr,
                       seq_len: tl.constexpr, features: tl.constexpr,
                       BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    for i in range(pid, seq_len, psize):
        row_x_offset = i * features
        sum_sq = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
        for j in range(0, features, BLOCK_SIZE):
            cols = j + tl.arange(0, BLOCK_SIZE)
            mask = cols < features
            x_val = tl.load(
                x_ptr + row_x_offset + cols, mask=mask,
                other=0.0).to(tl.float32)
            sum_sq += x_val * x_val
        rms_factor = tl.rsqrt(tl.sum(sum_sq) / features + 1e-6)
        for j in range(0, features, BLOCK_SIZE):
            cols = j + tl.arange(0, BLOCK_SIZE)
            mask = cols < features
            x_val = tl.load(
                x_ptr + row_x_offset + cols, mask=mask,
                other=0.0).to(tl.float32)
            x_norm = x_val * rms_factor
            s_scale = tl.load(
                style_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            s_shift = tl.load(
                style_ptr + features + cols, mask=mask,
                other=0.0).to(tl.float32)
            s_gate = tl.load(
                style_ptr + 2 * features + cols, mask=mask,
                other=0.0).to(tl.float32)
            output_val = x_norm * (1.0 + s_scale) + s_shift
            tl.store(
                normed_x_ptr + row_x_offset + cols,
                output_val.to(tl.bfloat16),
                mask=mask)
            tl.store(
                gate_ptr + row_x_offset + cols,
                s_gate.to(tl.bfloat16),
                mask=mask)
