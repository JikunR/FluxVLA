# Source: https://github.com/NVIDIA/Isaac-GR00T
# Branch: n1.5-release
# Path: gr00t/model/backbone/eagle2_hg_model/modeling_eagle2_5_vl.py
# Modified from https://github.com/dexmal/realtime-vla/blob/main/pi0_infer.py

import inspect
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint as cp
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.models.siglip.modeling_siglip import SiglipVisionModel

from fluxvla.engines.utils.overwatch import initialize_overwatch
from fluxvla.models.third_party_models.eagle2_hg_model.modeling_eagle2_5_vl import \
    Eagle2_5_VLPreTrainedModel  # noqa: E501
from fluxvla.ops import (combine_1536_1152_twopart, layer_norm_small_kernel,
                         matmul_512x1152x1152_twopart_bias_res,
                         matmul_small_bias, matmul_small_bias_gelu,
                         matmul_small_bias_res, matmul_small_bias_res_mod,
                         matmul_split_k, merge_split_k_bias_res)
from .configuration_eagle2_5_vl import Eagle2_5_VLConfig
from .radio_model import RADIOModel

overwatch = initialize_overwatch(__name__)

# copy from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava_onevision/modeling_llava_onevision.py#L241C1-L280C1  # noqa: E501
EAGLE2_5_VL_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the  # noqa: E501
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads  # noqa: E501
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.  # noqa: E501
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage  # noqa: E501
    and behavior.

    Parameters:
        config ([`Eagle2_5_VLConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not  # noqa: E501
            load the weights associated with the model, only the configuration. Check out the  # noqa: E501
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.  # noqa: E501
"""


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


@torch.compile
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
            hidden=1152,
        )
        combine_1536_1152_twopart[(256, )](
            inp_ptr=buf,
            out_ptr=out,
            seq_len=512,
            hidden=1152,
        )
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


class Eagle2_5_VLInferenceForConditionalGeneration(Eagle2_5_VLPreTrainedModel,
                                                   GenerationMixin):
    config_class = Eagle2_5_VLConfig

    def __init__(self,
                 config: Eagle2_5_VLConfig,
                 vision_model=None,
                 language_model=None):
        super().__init__(config)

        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        if config.use_pixel_shuffle:
            self.num_image_token = int(
                (image_size // patch_size)**2 * (config.downsample_ratio**2))
        else:
            self.num_image_token = int((image_size // patch_size)**2)

        self.select_layer = config.select_layer
        self.downsample_ratio = config.downsample_ratio
        self.loss_version = config.loss_version
        self.mlp_checkpoint = config.mlp_checkpoint
        self.use_pixel_shuffle = config.use_pixel_shuffle
        self.mlp_connector_layers = config.mlp_connector_layers
        overwatch.info(f'num_image_token: {self.num_image_token}')
        overwatch.info(f'mlp_checkpoint: {self.mlp_checkpoint}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            if config.vision_config.model_type == 'siglip_vision_model':
                config.vision_config._attn_implementation = 'flash_attention_2'
                self.vision_model = SiglipVisionModel(config.vision_config)
            elif config.vision_config.model_type == 'radio':
                self.vision_model = RADIOModel(config.vision_config)
            else:
                raise NotImplementedError(
                    f'{config.vision_config.model_type} is not implemented.')

        if language_model is not None:
            self.language_model = language_model
        else:
            if config.text_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.text_config)
            elif config.text_config.architectures[0] == 'Phi3ForCausalLM':
                raise NotImplementedError('Phi3 is not implemented.')
                # self.language_model = Phi3ForCausalLM(config.text_config)
            elif config.text_config.architectures[0] == 'Qwen2ForCausalLM':
                assert (
                    config.text_config._attn_implementation ==
                    'flash_attention_2'
                ), f'Qwen2 must use flash_attention_2 but got {config.text_config._attn_implementation}'  # noqa: E501
                self.language_model = Qwen2ForCausalLM(config.text_config)
            elif config.text_config.architectures[0] == 'Qwen3ForCausalLM':
                self.language_model = Qwen3ForCausalLM(config.text_config)
            else:
                raise NotImplementedError(
                    f'{config.text_config.architectures[0]} is not implemented.'  # noqa: E501
                )

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.text_config.hidden_size

        if config.mlp_connector_layers == 2:
            self.mlp1 = nn.Sequential(
                nn.LayerNorm(vit_hidden_size *
                             int(1 / self.downsample_ratio)**2),
                nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio)**2,
                          llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size),
            )
        elif config.mlp_connector_layers == 1 and config.use_pixel_shuffle:
            self.mlp1 = nn.Sequential(
                nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio)**2,
                          llm_hidden_size), )
        elif config.mlp_connector_layers == 1 and not config.use_pixel_shuffle:
            self.mlp1 = nn.Sequential(
                nn.Linear(vit_hidden_size, llm_hidden_size), )
        else:
            raise NotImplementedError(
                f'{config.mlp_connector_layers} is not implemented.')

        self.image_token_index = config.image_token_index
        self.neftune_alpha = None

        if config.use_backbone_lora:
            self.wrap_backbone_lora(
                r=config.use_backbone_lora,
                lora_alpha=2 * config.use_backbone_lora)

        self.use_llm_lora = config.use_llm_lora
        if config.use_llm_lora:
            self.wrap_llm_lora(
                r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora)

        self.check_forward_kwargs()

        self.weights = {
            'vision_patch_embedding_w':
            torch.empty(14, 14, 3, 1152, dtype=torch.bfloat16, device='cuda'),
            'vision_patch_embedding_b':
            torch.empty(1152, dtype=torch.bfloat16, device='cuda'),
            'vision_position_embedding':
            torch.empty(256, 1152, dtype=torch.bfloat16, device='cuda'),
            'vision_attn_qkv_w':
            torch.empty(
                27, 1152, 3 * 1152, dtype=torch.bfloat16, device='cuda'),
            'vision_attn_qkv_b':
            torch.empty(27, 3 * 1152, dtype=torch.bfloat16, device='cuda'),
            'vision_attn_o_w':
            torch.empty(27, 1152, 1152, dtype=torch.bfloat16, device='cuda'),
            'vision_attn_o_b':
            torch.empty(27, 1152, dtype=torch.bfloat16, device='cuda'),
            'vision_ffn_up_w':
            torch.empty(27, 1152, 4304, dtype=torch.bfloat16, device='cuda'),
            'vision_ffn_up_b':
            torch.empty(27, 4304, dtype=torch.bfloat16, device='cuda'),
            'vision_ffn_down_w':
            torch.empty(27, 4304, 1152, dtype=torch.bfloat16, device='cuda'),
            'vision_ffn_down_b':
            torch.empty(27, 1152, dtype=torch.bfloat16, device='cuda'),
            'vision_pre_attn_norm_w':
            torch.empty(27, 1152, dtype=torch.bfloat16, device='cuda'),
            'vision_pre_attn_norm_b':
            torch.empty(27, 1152, dtype=torch.bfloat16, device='cuda'),
            'vision_pre_ffn_norm_w':
            torch.empty(27, 1152, dtype=torch.bfloat16, device='cuda'),
            'vision_pre_ffn_norm_b':
            torch.empty(27, 1152, dtype=torch.bfloat16, device='cuda'),
            'vision_final_norm_w':
            torch.empty(1152, dtype=torch.bfloat16, device='cuda'),
            'vision_final_norm_b':
            torch.empty(1152, dtype=torch.bfloat16, device='cuda')
        }

        self.loaded_weights = False

    def check_forward_kwargs(self):
        # We intentionally avoid using **kwargs in forward because Hugging Face Transformers  # noqa: E501
        # has special handling for functions with **kwargs parameters that would affect  # noqa: E501
        # how our model is processed during training and inference.
        forward_params = inspect.signature(self.forward).parameters
        assert not any(k.kind == inspect.Parameter.VAR_KEYWORD
                       for k in forward_params.values())

    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=[
                'self_attn.q_proj',
                'self_attn.k_proj',
                'self_attn.v_proj',
                'self_attn.out_proj',
                'mlp.fc1',
                'mlp.fc2',
            ],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        self.vision_model.print_trainable_parameters()

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=[
                'self_attn.q_proj',
                'self_attn.k_proj',
                'self_attn.v_proj',
                'self_attn.o_proj',
                'mlp.gate_proj',
                'mlp.down_proj',
                'mlp.up_proj',
            ],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type='CAUSAL_LM',
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        self.language_model.print_trainable_parameters()
        self.use_llm_lora = True

    def _load_weights_and_buffer(self):
        self.weights['vision_patch_embedding_w'].copy_(
            self.vision_model.vision_model.embeddings.patch_embedding.weight.
            permute(2, 3, 1, 0))
        self.weights['vision_patch_embedding_b'].copy_(
            self.vision_model.vision_model.embeddings.patch_embedding.bias)
        self.weights['vision_position_embedding'].copy_(
            self.vision_model.vision_model.embeddings.position_embedding.weight
        )
        self.weights['vision_attn_qkv_w'].copy_(
            torch.stack([
                torch.cat([
                    self.vision_model.vision_model.encoder.layers[i].self_attn.
                    q_proj.weight, self.vision_model.vision_model.encoder.
                    layers[i].self_attn.k_proj.weight, self.vision_model.
                    vision_model.encoder.layers[i].self_attn.v_proj.weight
                ],
                          dim=0).permute(1, 0) for i in range(27)
            ]))
        self.weights['vision_attn_qkv_b'].copy_(
            torch.stack([
                torch.cat([
                    self.vision_model.vision_model.encoder.layers[i].self_attn.
                    q_proj.bias, self.vision_model.vision_model.encoder.
                    layers[i].self_attn.k_proj.bias, self.vision_model.
                    vision_model.encoder.layers[i].self_attn.v_proj.bias
                ],
                          dim=0) for i in range(27)
            ]))
        self.weights['vision_attn_o_w'].copy_(
            torch.stack([
                self.vision_model.vision_model.encoder.layers[i].self_attn.
                out_proj.weight.permute(1, 0) for i in range(27)
            ]))
        self.weights['vision_attn_o_b'].copy_(
            torch.stack([
                self.vision_model.vision_model.encoder.layers[i].self_attn.
                out_proj.bias for i in range(27)
            ]))
        self.weights['vision_ffn_up_w'].copy_(
            torch.stack([
                self.vision_model.vision_model.encoder.layers[i].mlp.fc1.weight
                for i in range(27)
            ]).permute(0, 2, 1))
        self.weights['vision_ffn_up_b'].copy_(
            torch.stack([
                self.vision_model.vision_model.encoder.layers[i].mlp.fc1.bias
                for i in range(27)
            ]))
        self.weights['vision_ffn_down_w'].copy_(
            torch.stack([
                self.vision_model.vision_model.encoder.layers[i].mlp.fc2.weight
                for i in range(27)
            ]).permute(0, 2, 1))
        self.weights['vision_ffn_down_b'].copy_(
            torch.stack([
                self.vision_model.vision_model.encoder.layers[i].mlp.fc2.bias
                for i in range(27)
            ]))
        self.weights['vision_pre_attn_norm_w'].copy_(
            torch.stack([
                self.vision_model.vision_model.encoder.layers[i].layer_norm1.
                weight for i in range(27)
            ]))
        self.weights['vision_pre_attn_norm_b'].copy_(
            torch.stack([
                self.vision_model.vision_model.encoder.layers[i].layer_norm1.
                bias for i in range(27)
            ]))
        self.weights['vision_pre_ffn_norm_w'].copy_(
            torch.stack([
                self.vision_model.vision_model.encoder.layers[i].layer_norm2.
                weight for i in range(27)
            ]))
        self.weights['vision_pre_ffn_norm_b'].copy_(
            torch.stack([
                self.vision_model.vision_model.encoder.layers[i].layer_norm2.
                bias for i in range(27)
            ]))
        self.weights['vision_final_norm_w'].copy_(
            self.vision_model.vision_model.post_layernorm.weight)
        self.weights['vision_final_norm_b'].copy_(
            self.vision_model.vision_model.post_layernorm.bias)
        num_views = 2
        self.buffers = {
            'observation_images_normalized':
            torch.empty(
                num_views, 224, 224, 3, dtype=torch.bfloat16, device='cuda'),
            'vision_x':
            torch.empty(
                num_views, 256, 1152, dtype=torch.bfloat16, device='cuda'),
            'vision_x_norm':
            torch.empty(
                num_views, 256, 1152, dtype=torch.bfloat16, device='cuda'),
            'vision_QKV':
            torch.empty(
                num_views, 256, 3 * 1152, dtype=torch.bfloat16, device='cuda'),
            'vision_hidden':
            torch.empty(
                num_views, 256, 4304, dtype=torch.bfloat16, device='cuda'),
            'vision_x_split_k_buf':
            torch.empty((num_views * 256 * 1152 * 4, ),
                        dtype=torch.float32,
                        device='cuda')
        }

        # Initialize vision CUDA Graph
        self.vision_graph = torch.cuda.CUDAGraph()
        self.record_vision_graph()

    def record_vision_run(self):
        """Core vision encoder computation for CUDA Graph capture."""
        conv2d_embed_n256_1152_res(
            self.buffers['observation_images_normalized'],
            self.weights['vision_patch_embedding_w'],
            self.weights['vision_patch_embedding_b'],
            self.weights['vision_position_embedding'],
            self.buffers['vision_x'])

        for i in range(27):
            layer_norm_QKV_matmul_n256_1152_3456_bias(
                self.buffers['vision_x'],
                self.weights['vision_pre_attn_norm_w'][i],
                self.weights['vision_pre_attn_norm_b'][i],
                self.weights['vision_attn_qkv_w'][i],
                self.weights['vision_attn_qkv_b'][i],
                self.buffers['vision_QKV'], self.buffers['vision_x_norm'])

            attn = AttnMultiKey(self.buffers['vision_QKV'])

            matmul_n256_1152_1152_bias_res(
                attn, self.weights['vision_attn_o_w'][i],
                self.weights['vision_attn_o_b'][i], self.buffers['vision_x'],
                self.buffers['vision_x'], self.buffers['vision_x_split_k_buf'])

            layer_norm_matmul_n256_1152_4304_bias_gelu(
                self.buffers['vision_x'],
                self.weights['vision_pre_ffn_norm_w'][i],
                self.weights['vision_pre_ffn_norm_b'][i],
                self.weights['vision_ffn_up_w'][i],
                self.weights['vision_ffn_up_b'][i],
                self.buffers['vision_hidden'], self.buffers['vision_x_norm'])

            matmul_n256_4304_1152_bias_res(
                self.buffers['vision_hidden'],
                self.weights['vision_ffn_down_w'][i],
                self.weights['vision_ffn_down_b'][i], self.buffers['vision_x'],
                self.buffers['vision_x'], self.buffers['vision_x_split_k_buf'])

    def record_vision_graph(self):
        """Record vision encoder computation into CUDA Graph."""
        # Warm-up runs
        for i in range(3):
            self.record_vision_run()
        # Capture CUDA Graph
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            self.vision_graph.capture_begin()
            self.record_vision_run()
            self.vision_graph.capture_end()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        num_tiles_list: Optional[List[torch.Tensor]] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if not self.loaded_weights:
            self._load_weights_and_buffer()
            self.loaded_weights = True

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # noqa: E501

        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        vit_embeds = self.extract_feature(pixel_values)

        if image_flags is not None:
            image_flags = image_flags.view(-1)
            vit_embeds = vit_embeds[image_flags == 1]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = input_ids == self.image_token_index
        try:
            input_embeds[
                selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(
                    -1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(
                f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '  # noqa: E501
                f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[
                selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(
                -1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits, ) + outputs[1:]
            return (loss, ) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)  # noqa: E501
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))

        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        # Convert from NCHW [B, C, H, W] to NHWC [B, H, W, C] for CUDA kernel
        self.buffers['observation_images_normalized'].copy_(
            pixel_values.permute(0, 2, 3, 1))

        # Replay CUDA Graph for vision encoder
        self.vision_graph.replay()

        # Post-processing (not in CUDA Graph due to checkpoint compatibility)
        vit_embeds = self.vision_model.vision_model.post_layernorm(
            self.buffers['vision_x'])

        if self.mlp_checkpoint and vit_embeds.requires_grad:
            vit_embeds = cp.checkpoint(self.mlp1, vit_embeds)
        else:
            vit_embeds = self.mlp1(vit_embeds)

        return vit_embeds

    @torch.no_grad()
    def generate(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        visual_features: Optional[torch.FloatTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        output_hidden_states: Optional[bool] = None,
        image_sizes: Optional[List[Tuple[int, int]]] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:

        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)

            input_embeds = self.language_model.get_input_embeddings()(
                input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = input_ids == self.config.image_token_index
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(
                input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(
                input_ids)

        if 'use_cache' not in generate_kwargs:
            generate_kwargs['use_cache'] = True

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            **generate_kwargs,
        )

        return outputs

    # Copied from transformers.models.llava_next.modeling_llava_next.LlavaNextForConditionalGeneration.get_input_embeddings  # noqa: E501
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    # Copied from transformers.models.llava_next.modeling_llava_next.LlavaNextForConditionalGeneration.set_input_embeddings  # noqa: E501
    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    # Copied from transformers.models.llava_next.modeling_llava_next.LlavaNextForConditionalGeneration.get_output_embeddings  # noqa: E501
    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    # Copied from transformers.models.llava_next.modeling_llava_next.LlavaNextForConditionalGeneration.set_output_embeddings  # noqa: E501
    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    # Copied from transformers.models.llava_next.modeling_llava_next.LlavaNextForConditionalGeneration.set_decoder  # noqa: E501
    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    # Copied from transformers.models.llava_next.modeling_llava_next.LlavaNextForConditionalGeneration.get_decoder  # noqa: E501
    def get_decoder(self):
        return self.language_model.get_decoder()
