# Cosmos3VLA 设计笔记

记录 `Cosmos3VLA` 集成过程中的关键设计决策及其依据。

______________________________________________________________________

## 1. Config 驱动的惯例

FluxVLA 的审美是 **config.py 即架构**：打开配置文件就能知道模型
规格，不需要打开 checkpoint 目录。

Cosmos3VLA 目前只有 `pretrained_name_or_path`，架构信息藏在
checkpoint 的 `config.json` 里，乍看不符合惯例。但对比后确认
这是合理的：`WanBackbone` 同样只传路径、不展开 Wan2.1-14B 的
层数参数，因为它是"借用的大底座，不改结构"。Qwen3-VL backbone
同理。

action 侧参数（`max_action_dim`、`num_embodiment_domains`）已在
config.py 显式写出，这是 FluxVLA 真正关心的部分。

### Nano vs Super 参数对比

查询 [nvidia/Cosmos3-Super](https://huggingface.co/nvidia/Cosmos3-Super)
的 `config.json` 后确认：

| 参数                            | Nano                   | Super        |
| ------------------------------- | ---------------------- | ------------ |
| VLM backbone                    | Qwen3-VL-8B            | Qwen3-VL-32B |
| `text_config.hidden_size`       | 4096                   | 5120         |
| `text_config.num_hidden_layers` | 36                     | 64           |
| `max_action_dim`                | **64**                 | **64**       |
| `num_embodiment_domains`        | **32**                 | **32**       |
| VAE                             | Wan2.2 4×16×16         | 完全相同     |
| RF shift                        | `{256:3,480:5,720:10}` | 完全相同     |

**action 侧参数两者完全相同。** 因此 `pretrained_name_or_path`
就是完整的架构描述符，支持 Super 只需新建一个配置文件换路径，
代码零修改。

______________________________________________________________________

## 2. 架构类比：MoT 与 π₀

Cosmos3 是 **Mixture-of-Transformers（MoT）**，每个 `MoTDecoderLayer`
内部有两套完全独立的权重：

- **Understanding tower**：`q_proj, k_proj, v_proj, o_proj, mlp, layernorm`（无后缀）
- **Generation tower**：`q_proj_moe_gen, k_proj_moe_gen, ..., mlp_moe_gen`（`_moe_gen` 后缀）

注意力方向通过 `two_way` attention 实现：

- Understanding token：causal attention，看不到 generation token（action/video/audio）
- Generation token：full attention，可以 attend 所有 token（包括 understanding）

这和 **π₀** 在概念上完全同构：

|                    | π₀                               | Cosmos3 MoT              |
| ------------------ | -------------------------------- | ------------------------ |
| Understanding 权重 | 独立 `nn.Module`（PaliGemma）    | 每层无后缀参数           |
| Generation 权重    | 独立 `nn.Module`（Gemma Expert） | 每层 `_moe_gen` 参数     |
| 单向性             | 两个 module 跑完后 cross-attend  | `two_way` attention mask |
| Generation 目标    | action only                      | action + video + audio   |

**物理封装的区别**：π₀ 的两条路径是两个分开的 `nn.Module`，可以
在 config 里各自展开架构参数（π₀ 的 `llm_expert` 有独立的
`hidden_size=1024`）；Cosmos3 的两条路径交织在同一个
`Cosmos3VFMNetwork` 里，generation tower 和 understanding tower
共用相同的 `hidden_size`/`num_layers`，没有可分割的模块边界。

### 是否按 π₀ 拆 `vlm_backbone + vla_head`？

待定。形式上可以拆，`vlm_backbone` 对应 understanding tower 规格
（路径），`vla_head` 对应 generation tower 规格。但 generation
tower 没有独立的层数/宽度（与 understanding tower 完全共享），
`vla_head` 里能写的有意义参数只有 `max_action_dim=64`
和 `num_embodiment_domains=32`，信息量有限。

当前保持扁平结构，等 Cosmos3-Super 实际使用时再按需决策。

______________________________________________________________________

## 3. 相关文件

| 文件                                                                                                          | 说明                           |
| ------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| [fluxvla/models/vlas/cosmos3_vla.py](../fluxvla/models/vlas/cosmos3_vla.py)                                   | 主模型类                       |
| [fluxvla/collators/cosmos3_collator.py](../fluxvla/collators/cosmos3_collator.py)                             | Collator                       |
| [fluxvla/transforms/transform_cosmos3.py](../fluxvla/transforms/transform_cosmos3.py)                         | 数据变换                       |
| [fluxvla/engines/runners/cosmos3_inference_runner.py](../fluxvla/engines/runners/cosmos3_inference_runner.py) | 推理 Runner                    |
| [fluxvla/models/third_party_models/cosmos3/](../fluxvla/models/third_party_models/cosmos3/)                   | vendored cosmos-framework 子集 |
| [configs/cosmos3nano/](../configs/cosmos3nano/)                                                               | 训练配置（HUD04 + Libero）     |
