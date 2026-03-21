"""
PI05 Triton Inference — single CUDA Graph for the entire pipeline.

Combines vision_encoder + transformer_encoder + transformer_decoder into
ONE CUDA Graph, eliminating inter-graph overhead.

Architecture inference pattern:
  - L1: raw Triton kernels imported from ``fluxvla.ops``
  - L2: composed operations (dimension-bound wrappers)
  - L3: model functions (vision_encoder, transformer_encoder, etc.)
  - L4: ``PI05Inference`` inference class

Usage::

    model = build_vla_from_cfg(cfg.model)
    model.load_state_dict(...)
    patch_with_unified_triton(model, num_views=2, max_prompt_len=48,
                              chunk_size=10, num_steps=10)
"""
import torch
import torch.nn as nn

from .attention_triton_ops import (matmul_abT_scale,
                                   matmul_n_2048_2560_qkv_rope,
                                   matmul_rope_qkv, scaled_matmul_rope_qkv,
                                   softmax_kernel_masklen,
                                   softmax_kernel_prefix_suffix)
# yapf: disable
from .matmul_triton_ops import (combine_1536_1152_twopart,
                                matmul_512x1152x1152_twopart_bias_res,
                                matmul_small, matmul_small_bias,
                                matmul_small_bias_gelu, matmul_small_bias_res,
                                matmul_small_bias_res_mod,
                                matmul_small_bias_silu, matmul_small_gate,
                                matmul_small_res, matmul_small_res_gate,
                                matmul_split_k, merge_split_k_bias_res)
# yapf: enable
from .norm_triton_ops import (adarms_norm_kernel, layer_norm_small_kernel,
                              rms_norm_kernel, rmsnorm_factor_kernel)


def conv2d_embed_n256_1152_res(images, patch_w, patch_b, pos_emb, out):
    nviews = images.shape[0]
    img_input = images.view(nviews, 16, 14, 16, 14,
                            3).permute(0, 1, 3, 2, 4, 5).contiguous()
    matmul_small_bias_res_mod[(256 * nviews // 64) * (1152 // 64), ](
        img_input,
        patch_w,
        out,
        patch_b,
        pos_emb,
        seq_len=256 * nviews,
        features=3 * 14 * 14,
        hidden=1152,
        i_mod=256,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_K=32)


def layer_norm_QKV_matmul_n256_1152_3456_bias(x, norm_w, norm_b, qkv_w, qkv_b,
                                              out, x_norm):
    num_views = x.shape[0]
    seq_len = 256 * num_views
    layer_norm_small_kernel[seq_len, ](
        x, x_norm, norm_w, norm_b, seq_len=seq_len, features=1152)
    matmul_small_bias[((seq_len + 63) // 64) * (3456 // 64), ](
        x_norm,
        qkv_w,
        out,
        qkv_b,
        seq_len=seq_len,
        features=1152,
        hidden=3456,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_K=32)


# @torch.compile
def AttnMultiKey(QKV):
    QKV = QKV.view(-1, 256, 3, 16, 72).permute(0, 2, 3, 1, 4)
    Q = QKV[:, 0]
    K = QKV[:, 1]
    V = QKV[:, 2]
    attn = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    attn = attn.transpose(1, 2).reshape(Q.shape[0], 256, 1152)
    return attn


def matmul_n256_1152_1152_bias_res(x, weight, bias, res, out, buf):
    num_views = x.shape[0]
    if num_views == 2:
        matmul_512x1152x1152_twopart_bias_res[(256, )](
            old_ptr=res,
            inp_ptr=x,
            weight_ptr=weight,
            bias_ptr=bias,
            out_ptr=out,
            out2_ptr=buf,
            seq_len=512,
            features=1152,
            hidden=1152)
        combine_1536_1152_twopart[(256, )](
            inp_ptr=buf, out_ptr=out, seq_len=512, hidden=1152)
        return
    seq_len = 256 * num_views
    matmul_small_bias_res[((seq_len + 31) // 32) * (1152 // 64), ](
        x,
        weight,
        out,
        bias,
        res,
        seq_len=seq_len,
        features=1152,
        hidden=1152,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_K=32)


def layer_norm_matmul_n256_1152_4304_bias_gelu(x, norm_w, norm_b, weight, bias,
                                               out, x_norm):
    num_views = x.shape[0]
    seq_len = 256 * num_views
    layer_norm_small_kernel[seq_len, ](
        x, x_norm, norm_w, norm_b, seq_len=seq_len, features=1152)
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_K = 64
    matmul_small_bias_gelu[((seq_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N) *
                           ((4304 + (BLOCK_SIZE_M - 1)) // BLOCK_SIZE_M), ](
                               x_norm,
                               weight,
                               out,
                               bias,
                               seq_len=seq_len,
                               features=1152,
                               hidden=4304,
                               BLOCK_SIZE_N=BLOCK_SIZE_N,
                               BLOCK_SIZE_M=BLOCK_SIZE_M,
                               BLOCK_SIZE_K=BLOCK_SIZE_K)


def matmul_n256_4304_1152_bias_res(x, weight, bias, res, out, buf):
    num_views = x.shape[0]
    seq_len = 256 * num_views
    matmul_split_k[((seq_len + 64) // 64) * (1152 // 64) * 4, ](
        x,
        weight,
        buf,
        seq_len=seq_len,
        features=4304,
        hidden=1152,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_K=64,
        SPLIT_K=4)
    merge_split_k_bias_res[(seq_len * 1152 + 1023) // 1024, ](
        buf, bias, res, out, seq_len=seq_len, hidden=1152, SPLIT_K=4)


def layer_norm_matmul_n256_1152_2048_bias(x, norm_w, norm_b, proj_w, proj_b,
                                          out, x_norm):
    seq_len = x.shape[0] * 256
    layer_norm_small_kernel[seq_len, ](
        x, x_norm, norm_w, norm_b, seq_len=seq_len, features=1152, eps=1e-5)
    matmul_small_bias[((seq_len + 63) // 64) * (2048 // 64), ](
        x_norm,
        proj_w,
        out,
        proj_b,
        seq_len=seq_len,
        features=1152,
        hidden=2048,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_K=64)


def rms_matmul_n_2048_2560_qkv_rope(x, weight_qkv, rope_weight, Q, K, V,
                                    x_norm):
    seq_len = x.shape[0]
    rms_norm_kernel[(seq_len, )](x, x_norm, seq_len, 2048)
    matmul_n_2048_2560_qkv_rope[((seq_len + 63) // 64,
                                 2560 // 64)](x_norm, weight_qkv, rope_weight,
                                              Q, K, V, seq_len, 2048, 256, 8)


def matmul_n_2048_2048_res(x, weight, out):
    seq_len = x.shape[0]
    BLOCK_SIZE_N = 128
    if seq_len < 512:
        BLOCK_SIZE_N = 64
    matmul_small_res[((seq_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N) *
                     (2048 // 64), ](
                         x,
                         weight,
                         out,
                         out,
                         seq_len=seq_len,
                         features=2048,
                         hidden=2048,
                         BLOCK_SIZE_N=BLOCK_SIZE_N,
                         BLOCK_SIZE_M=64,
                         BLOCK_SIZE_K=64)


def rms_matmul_n_2048_16384_gate(x, weight1, weight2, out, x_norm):
    seq_len = x.shape[0]
    rms_norm_kernel[(seq_len, )](x, x_norm, seq_len, 2048)
    matmul_small_gate[((seq_len + 127) // 128,
                       (16384 + 63) // 64)](x_norm, weight1, weight2, out,
                                            seq_len, 2048, 16384)


def matmul_n_16384_2048_res(x, weight, out):
    seq_len = x.shape[0]
    BLOCK_SIZE_N = 128
    if seq_len < 512:
        BLOCK_SIZE_N = 64
    matmul_small_res[((seq_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N) *
                     (2048 // 64), ](
                         x,
                         weight,
                         out,
                         out,
                         seq_len=seq_len,
                         features=16384,
                         hidden=2048,
                         BLOCK_SIZE_N=BLOCK_SIZE_N,
                         BLOCK_SIZE_M=64,
                         BLOCK_SIZE_K=64)


def matmul_k8_n_256(x, V, out):
    total_queries = x.shape[0]
    total_keys = V.shape[0]
    head_dim = 256
    matmul_small[((total_keys + 31) // 32) * (head_dim // 32), ](
        x,
        V,
        out,
        seq_len=total_queries,
        features=total_keys,
        hidden=head_dim,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_K=64)


def matmul_1_1024_1024_bias_silu(x, weight, bias, out):
    seq_len = x.shape[0]
    matmul_small_bias_silu[((seq_len + 31) // 32) * (1024 // 32), ](
        x,
        weight,
        out,
        bias,
        seq_len=seq_len,
        features=1024,
        hidden=1024,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_K=64)


def matmul_k_32_1024_bias(x, weight, bias, out):
    seq_len = x.shape[0]
    matmul_small_bias[((seq_len + 31) // 32) * (1024 // 32), ](
        x,
        weight,
        out,
        bias,
        seq_len=seq_len,
        features=32,
        hidden=1024,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_K=32)


def adarms_norm_style_proj(x, time_emb, mod_w, mod_b, x_normed, gate, style):
    seq_len = x.shape[0]
    matmul_small_bias[((seq_len + 31) // 32) * (3072 // 32), ](
        time_emb,
        mod_w,
        style,
        mod_b,
        seq_len=seq_len,
        features=1024,
        hidden=3072,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_K=32)
    adarms_norm_kernel[(seq_len, )](
        x,
        style,
        x_normed,
        gate,
        seq_len=seq_len,
        features=1024,
        BLOCK_SIZE=512)


def matmul_k_1024_2560_qkv_rope(x_normed, weight_qkv, rope_weight, Q, K, V):
    seq_len = x_normed.shape[0]
    matmul_rope_qkv[(128, )](x_normed, seq_len, 1024, 256, 8, weight_qkv,
                             rope_weight, Q, K, V)


def rms_matmul_k_1024_2560_qkv_rope(x, weight_qkv, rope_weight, Q, K, V,
                                    x_norm_factor):
    seq_len = x.shape[0]
    rmsnorm_factor_kernel[(128, )](
        x, x_norm_factor, seq_len, 1024, eps=1e-6, BLOCK_SIZE=1024)
    scaled_matmul_rope_qkv[(128, )](x, x_norm_factor, seq_len, 1024, 256, 8,
                                    weight_qkv, rope_weight, Q, K, V)


def matmul_k_2048_1024_gate(x, weight, out, gate):
    seq_len = x.shape[0]
    matmul_small_res_gate[(128, )](
        x,
        weight,
        out,
        out,
        gate,
        seq_len=seq_len,
        features=2048,
        hidden=1024,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_K=128)


def matmul_k_4096_1024_gate(x, weight, out, gate):
    seq_len = x.shape[0]
    matmul_small_res_gate[(((seq_len + 15) // 16) * (1024 // 32), )](
        x,
        weight,
        out,
        out,
        gate,
        seq_len=seq_len,
        features=4096,
        hidden=1024,
        BLOCK_SIZE_N=16,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_K=256)


def vision_encoder(weights, buffers, num_views):
    conv2d_embed_n256_1152_res(buffers['observation_images_normalized'],
                               weights['vision_patch_embedding_w'],
                               weights['vision_patch_embedding_b'],
                               weights['vision_position_embedding'],
                               buffers['vision_x'])

    for i in range(27):
        layer_norm_QKV_matmul_n256_1152_3456_bias(
            buffers['vision_x'], weights['vision_pre_attn_norm_w'][i],
            weights['vision_pre_attn_norm_b'][i],
            weights['vision_attn_qkv_w'][i], weights['vision_attn_qkv_b'][i],
            buffers['vision_QKV'], buffers['vision_x_norm'])

        attn = AttnMultiKey(buffers['vision_QKV'])

        matmul_n256_1152_1152_bias_res(attn, weights['vision_attn_o_w'][i],
                                       weights['vision_attn_o_b'][i],
                                       buffers['vision_x'],
                                       buffers['vision_x'],
                                       buffers['vision_x_split_k_buf'])

        layer_norm_matmul_n256_1152_4304_bias_gelu(
            buffers['vision_x'], weights['vision_pre_ffn_norm_w'][i],
            weights['vision_pre_ffn_norm_b'][i], weights['vision_ffn_up_w'][i],
            weights['vision_ffn_up_b'][i], buffers['vision_hidden'],
            buffers['vision_x_norm'])

        matmul_n256_4304_1152_bias_res(buffers['vision_hidden'],
                                       weights['vision_ffn_down_w'][i],
                                       weights['vision_ffn_down_b'][i],
                                       buffers['vision_x'],
                                       buffers['vision_x'],
                                       buffers['vision_x_split_k_buf'])


def transformer_encoder(weights, buffers, encoder_seq_len):
    layer_norm_matmul_n256_1152_2048_bias(
        buffers['vision_x'], weights['vision_final_norm_w'],
        weights['vision_final_norm_b'],
        weights['encoder_multi_modal_projector_w'],
        weights['encoder_multi_modal_projector_b'], buffers['encoder_x'],
        buffers['vision_x_norm'])

    for i in range(18):
        rms_matmul_n_2048_2560_qkv_rope(
            buffers['encoder_x'], weights['encoder_attn_qkv_w'][i],
            buffers['encoder_rope_weights'], buffers['encoder_Q'],
            buffers['encoder_K'][i, :encoder_seq_len],
            buffers['encoder_V'][i, :encoder_seq_len],
            buffers['encoder_x_norm'])

        if i != 17:
            scale = 1.0 / (256**0.5)
            total_queries = buffers['encoder_Q'].shape[0]
            total_keys = encoder_seq_len
            matmul_abT_scale[(((total_queries + 31) // 32) *
                              ((total_keys + 31) // 32), )](
                                  buffers['encoder_Q'],
                                  buffers['encoder_K'][i, :encoder_seq_len],
                                  buffers['encoder_logits_buf'],
                                  total_queries,
                                  total_keys,
                                  256,
                                  scale,
                                  BLOCK_SIZE_M=32,
                                  BLOCK_SIZE_N=32,
                                  BLOCK_SIZE_K=64)

            softmax_kernel_masklen[((total_queries + 3) // 4, )](
                buffers['encoder_logits_buf'],
                total_queries,
                total_keys,
                buffers['valid_encoder_len'],
                buffers['encoder_attn_buf'],
                BLOCK_SIZE_M=4,
                BLOCK_SIZE=1024)

            matmul_k8_n_256(buffers['encoder_attn_buf'],
                            buffers['encoder_V'][i, :encoder_seq_len],
                            buffers['encoder_ctx_buf'])

            matmul_n_2048_2048_res(buffers['encoder_ctx_buf'].view(-1, 2048),
                                   weights['encoder_attn_o_w'][i],
                                   buffers['encoder_x'])

            rms_matmul_n_2048_16384_gate(buffers['encoder_x'],
                                         weights['encoder_ffn_gate_w'][i],
                                         weights['encoder_ffn_up_w'][i],
                                         buffers['encoder_hidden'],
                                         buffers['encoder_x_norm'])

            matmul_n_16384_2048_res(buffers['encoder_hidden'],
                                    weights['encoder_ffn_down_w'][i],
                                    buffers['encoder_x'])


def transformer_decoder(weights, buffers, encoder_seq_len, num_steps=10):
    for step in range(num_steps):
        matmul_1_1024_1024_bias_silu(
            weights['decoder_time_embeds'][step].view(1, -1),
            weights['decoder_time_mlp_in_w'], weights['decoder_time_mlp_in_b'],
            buffers['decoder_x_buf'])
        matmul_1_1024_1024_bias_silu(buffers['decoder_x_buf'],
                                     weights['decoder_time_mlp_out_w'],
                                     weights['decoder_time_mlp_out_b'],
                                     buffers['decoder_time_emb'])
        matmul_k_32_1024_bias(buffers['diffusion_noise'],
                              weights['decoder_action_in_proj_w'],
                              weights['decoder_action_in_proj_b'],
                              buffers['decoder_x'])
        seq_len = buffers['decoder_x'].shape[0]

        for i in range(18):
            adarms_norm_style_proj(buffers['decoder_x'],
                                   buffers['decoder_time_emb'],
                                   weights['decoder_pre_attn_norm_mod_w'][i],
                                   weights['decoder_pre_attn_norm_mod_b'][i],
                                   buffers['x_normed_buf'],
                                   buffers['gate_buf'],
                                   buffers['decoder_style'])

            matmul_k_1024_2560_qkv_rope(
                buffers['x_normed_buf'], weights['decoder_attn_qkv_w'][i],
                buffers['decoder_rope_weights'], buffers['decoder_q_buf'],
                buffers['encoder_K'][i, encoder_seq_len:encoder_seq_len +
                                     seq_len],
                buffers['encoder_V'][i, encoder_seq_len:encoder_seq_len +
                                     seq_len])

            total_queries = buffers['decoder_q_buf'].shape[0]
            prefix_keys = encoder_seq_len
            suffix_keys = seq_len
            total_keys = prefix_keys + suffix_keys

            matmul_abT_scale[(((total_queries + 31) // 32) *
                              ((total_keys + 31) // 32), )](
                                  buffers['decoder_q_buf'],
                                  buffers['encoder_K'][i, :encoder_seq_len +
                                                       seq_len],
                                  buffers['decoder_logits_buf'],
                                  total_queries,
                                  total_keys,
                                  256,
                                  256**-0.5,
                                  BLOCK_SIZE_M=32,
                                  BLOCK_SIZE_N=32,
                                  BLOCK_SIZE_K=64)

            softmax_kernel_prefix_suffix[((total_queries + 3) // 4, )](
                buffers['decoder_logits_buf'],
                total_queries,
                prefix_keys,
                suffix_keys,
                buffers['valid_encoder_len'],
                buffers['decoder_attn_buf'],
                BLOCK_SIZE_M=4,
                BLOCK_SIZE=1024)

            matmul_k8_n_256(
                buffers['decoder_attn_buf'],
                buffers['encoder_V'][i, :encoder_seq_len + seq_len],
                buffers['decoder_q_buf'])

            matmul_k_2048_1024_gate(buffers['decoder_q_buf'].view(-1, 2048),
                                    weights['decoder_attn_o_w'][i],
                                    buffers['decoder_x'], buffers['gate_buf'])

            adarms_norm_style_proj(buffers['decoder_x'],
                                   buffers['decoder_time_emb'],
                                   weights['decoder_pre_ffn_norm_mod_w'][i],
                                   weights['decoder_pre_ffn_norm_mod_b'][i],
                                   buffers['x_normed_buf'],
                                   buffers['gate_buf'],
                                   buffers['decoder_style'])

            seq_len = buffers['decoder_x'].shape[0]
            matmul_small_gate[((seq_len + 127) // 128, (4096 + 63) // 64)](
                buffers['x_normed_buf'], weights['decoder_ffn_gate_w'][i],
                weights['decoder_ffn_up_w'][i], buffers['decoder_hidden'],
                seq_len, 1024, 4096)

            matmul_k_4096_1024_gate(buffers['decoder_hidden'],
                                    weights['decoder_ffn_down_w'][i],
                                    buffers['decoder_x'], buffers['gate_buf'])

        seq_len = buffers['decoder_x'].shape[0]
        adarms_norm_style_proj(buffers['decoder_x'],
                               buffers['decoder_time_emb'],
                               weights['decoder_final_norm_mod_w'],
                               weights['decoder_final_norm_mod_b'],
                               buffers['x_normed_buf'], buffers['gate_buf'],
                               buffers['decoder_style'])

        matmul_small_bias[((seq_len + 15) // 16) * (32 // 16), ](
            buffers['x_normed_buf'],
            weights['decoder_action_out_proj_w'],
            buffers['decoder_action_buf'],
            weights['decoder_action_out_proj_b'],
            seq_len=seq_len,
            features=1024,
            hidden=32,
            BLOCK_SIZE_N=16,
            BLOCK_SIZE_M=16,
            BLOCK_SIZE_K=256)

        buffers['diffusion_noise'].add_(
            buffers['decoder_action_buf'], alpha=-1.0 / num_steps)


def pi05_model(weights, buffers, num_views, encoder_seq_len, num_steps=10):
    vision_encoder(weights, buffers, num_views)
    transformer_encoder(weights, buffers, encoder_seq_len)
    transformer_decoder(weights, buffers, encoder_seq_len, num_steps)


class PI05Inference(nn.Module):
    """Triton inference for PI05 — single CUDA Graph.

    Records ``vision_encoder + transformer_encoder + transformer_decoder``
    into one CUDA Graph.
    """

    def __init__(self,
                 model,
                 num_views,
                 max_prompt_len,
                 chunk_size,
                 num_steps=10):
        super().__init__()
        self.num_views = num_views
        self.max_prompt_len = max_prompt_len
        self.chunk_size = chunk_size
        self.num_steps = num_steps
        self.num_encoder_layers = len(model.llm_backbone.layers)
        self.encoder_seq_len = num_views * 256 + max_prompt_len
        self.decoder_seq_len = chunk_size

        self.weights = {}
        self.bufs = {}

        # self._init_buffers()
        # self._init_rope_table()

        self._cuda_graph = None
        self._cuda_graph_ready = False

    @classmethod
    def from_weights(cls,
                     weights,
                     num_views,
                     max_prompt_len,
                     chunk_size,
                     num_steps=10):
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)

        obj.weights = weights
        obj.num_views = num_views
        obj.max_prompt_len = max_prompt_len
        obj.chunk_size = chunk_size
        obj.num_steps = num_steps
        obj.num_encoder_layers = 18
        obj.encoder_seq_len = num_views * 256 + max_prompt_len
        obj.decoder_seq_len = chunk_size
        obj.bufs = {}

        obj._init_buffers()
        obj._init_rope_table()

        obj._cuda_graph = None
        obj._cuda_graph_ready = False

        return obj

    def _init_buffers(self):
        nv = self.num_views
        enc = self.encoder_seq_len
        dec = self.decoder_seq_len
        bf = torch.bfloat16
        dev = 'cuda'

        self.bufs = {
            'observation_images_normalized':
            torch.zeros(nv, 224, 224, 3, dtype=bf, device=dev),
            'vision_x':
            torch.zeros(nv, 256, 1152, dtype=bf, device=dev),
            'vision_x_norm':
            torch.zeros(nv, 256, 1152, dtype=bf, device=dev),
            'vision_QKV':
            torch.zeros(nv, 256, 3 * 1152, dtype=bf, device=dev),
            'vision_hidden':
            torch.zeros(nv, 256, 4304, dtype=bf, device=dev),
            'vision_x_split_k_buf':
            torch.zeros(nv * 256 * 1152 * 4, dtype=torch.float32, device=dev),
            'encoder_rope_weights':
            torch.zeros(enc, 256, dtype=bf, device=dev),
            'encoder_x':
            torch.zeros(enc, 2048, dtype=bf, device=dev),
            'encoder_x_norm':
            torch.zeros(enc, 2048, dtype=bf, device=dev),
            'encoder_K':
            torch.zeros(18, enc + dec, 256, dtype=bf, device=dev),
            'encoder_V':
            torch.zeros(18, enc + dec, 256, dtype=bf, device=dev),
            'encoder_Q':
            torch.zeros(enc * 8, 256, dtype=bf, device=dev),
            'encoder_hidden':
            torch.zeros(enc, 16384, dtype=bf, device=dev),
            'valid_encoder_len':
            torch.zeros((1, ), dtype=torch.int32, device=dev),
            'encoder_logits_buf':
            torch.zeros(enc * 8, enc, dtype=torch.float32, device=dev),
            'encoder_attn_buf':
            torch.zeros(enc * 8, enc, dtype=bf, device=dev),
            'encoder_ctx_buf':
            torch.zeros(enc * 8, 256, dtype=bf, device=dev),
            'decoder_rope_weights':
            torch.zeros(dec, 256, dtype=bf, device=dev),
            'decoder_x':
            torch.zeros(dec, 1024, dtype=bf, device=dev),
            'decoder_x_buf':
            torch.zeros(dec, 1024, dtype=bf, device=dev),
            'decoder_action_buf':
            torch.zeros(dec, 32, dtype=bf, device=dev),
            'decoder_time_emb':
            torch.zeros(dec, 1024, dtype=bf, device=dev),
            'decoder_style':
            torch.zeros(dec, 1024 * 3, dtype=bf, device=dev),
            'decoder_norm_factor_buf':
            torch.zeros(dec, dtype=bf, device=dev),
            'decoder_q_buf':
            torch.zeros(dec * 8, 256, dtype=bf, device=dev),
            'decoder_logits_buf':
            torch.zeros(dec * 8, enc + dec, dtype=torch.float32, device=dev),
            'decoder_attn_buf':
            torch.zeros(dec * 8, enc + dec, dtype=bf, device=dev),
            'decoder_hidden':
            torch.zeros(dec, 4096, dtype=bf, device=dev),
            'decode_split_k_buf':
            torch.zeros(2, dec, 1024, dtype=torch.float32, device=dev),
            'x_normed_buf':
            torch.zeros(dec, 1024, dtype=bf, device=dev),
            'gate_buf':
            torch.zeros(dec, 1024, dtype=bf, device=dev),
            'diffusion_noise':
            torch.zeros(dec, 32, dtype=bf, device=dev),
        }

    def _init_rope_table(self):
        prefix_alloc = self.num_views * 256 + self.max_prompt_len
        max_pos = prefix_alloc - 1 + self.chunk_size
        position_ids = torch.arange(max_pos + 1, device='cuda')
        inv_freq = 1.0 / (10000**(
            torch.arange(0, 256, 2, dtype=torch.float32, device='cuda') / 256))
        k_phase = inv_freq[None, :] * position_ids[:, None]
        k_cos = torch.cos(k_phase).to(torch.bfloat16)
        k_sin = torch.sin(k_phase).to(torch.bfloat16)
        self._rope_table = torch.cat([k_cos[:, :, None], k_sin[:, :, None]],
                                     2).view(-1, 256)
        self.bufs['encoder_rope_weights'].copy_(
            self._rope_table[:prefix_alloc])

    def _get_decoder_rope_weights(self, prompt_len):
        start = self.num_views * 256 + prompt_len - 1
        end = start + self.chunk_size
        return self._rope_table[start:end]

    def _run_forward(self):
        pi05_model(self.weights, self.bufs, self.num_views,
                   self.encoder_seq_len, self.num_steps)

    def _build_cuda_graph(self):
        print('[Triton Inference] Recording CUDA Graph ...')
        for _ in range(3):
            self._run_forward()
        torch.cuda.synchronize()

        self._cuda_graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            self._cuda_graph.capture_begin()
            self._run_forward()
            self._cuda_graph.capture_end()
        torch.cuda.synchronize()

        self._cuda_graph_ready = True
        print('[Triton Inference] CUDA Graph recorded successfully!')

    def forward(self, images_nhwc, prompt_embeds, prompt_len, diffusion_noise):
        """Run the full unified Triton inference pipeline.

        Args:
            images_nhwc: images in [num_views, H, W, C] bfloat16 format.
            prompt_embeds: language embeddings [prompt_len, 2048] bfloat16.
            prompt_len: actual prompt token count (int).
            diffusion_noise: initial noise [chunk_size, 32] bfloat16.

        Returns:
            Denoised actions [chunk_size, 32] bfloat16.
        """
        self.bufs['observation_images_normalized'].copy_(images_nhwc)
        start = self.num_views * 256
        self.bufs['encoder_x'][start:start + prompt_len].copy_(prompt_embeds)
        self.bufs['valid_encoder_len'].fill_(start + prompt_len)
        self.bufs['decoder_rope_weights'].copy_(
            self._get_decoder_rope_weights(prompt_len))
        self.bufs['diffusion_noise'].copy_(diffusion_noise)

        if not self._cuda_graph_ready:
            self._build_cuda_graph()

        self._cuda_graph.replay()
        return self.bufs['diffusion_noise']
