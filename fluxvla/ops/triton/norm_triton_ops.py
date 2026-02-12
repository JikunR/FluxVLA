# Modified from https://github.com/dexmal/realtime-vla/blob/main/pi0_infer.py

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
