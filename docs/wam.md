# WAM 最小引入说明

本分支现在只保留 WAM 需要的最小 surface, 不再保留完整
FastWAM variant zoo。

## 保留内容

1. `fluxvla/models/vlas/wam_vla.py`

   只注册 `WAMVLA`。它负责按 FluxVLA 传统方式组装
   `vlm_backbone + video_latent_codec + vla_head`: `vlm_backbone` 只表示
   VLM/LLM context 侧, `video_latent_codec` 单独表示 video VAE, `vla_head`
   内显式声明 `video_expert + action_expert` 和 MoT。

2. `fluxvla/models/backbones/vlms/wan22_text_backbone.py`

   `Wan22TextBackbone` 是普通 VLM/text backbone, 只持有 Wan2.2 UMT5
   text encoder。它实现标准 `vlm_backbone.forward(images, lang_tokens, img_masks, lang_masks, ...)` 入口, 对 Wan text-only 场景会忽略 images,
   并把 `lang_tokens/lang_masks` 编成 WAM 使用的 `context/context_mask`。
   tokenizer 不归它持有。

   VAE 不再挂在 `vlm_backbone` 下面, 而是在模型顶层独立配置:
   `video_latent_codec=dict(type='Wan22VAE', ...)`。未来切 QwenVL 或其他
   LLM/VLM 时, 只需要替换 `vlm_backbone`; 未来切视频 latent codec 时,
   只替换 `video_latent_codec`。

3. `fluxvla/models/heads/wam_head.py`

   只注册 `WAMHead`, 不再暴露
   `FastWAMHead / FastWAMJointHead / FastWAMIDMHead`。`video_expert`,
   `action_expert` 和 MoT 都在 head 内部按 config build, `WAMVLA`
   只负责把 `video_latent_codec.temporal_downsample_factor` 等跨模块信息
   注入到 head config。

   `WAMModeCollator` 通过 `mode_probs` 为每个 local batch 采样一个
   `training_mode`, `WAMHead.forward()` 只消费该字段并计算对应目标的 loss。
   这样仍然能在训练步之间混合四种目标, 同时让单个 local batch 保持同一种
   attention mask, 避免 per-sample mixed mode 触发 4D mask 后拖慢
   FlashAttention 路径。

   - `forward`: 给定 action, 学 future video latent。
   - `idm`: 给定完整或扰动 video latent, 学 action。
   - `policy`: 给定 first frame, 学 action。
   - `joint`: 同时 denoise video latent 和 action。

4. `configs/wam/`

   canonical 配置只剩 WAM:

   - `wam_libero_object_full_finetune.py`
   - `wam_libero_10_full_finetune.py`
   - `wam_qwen3vl_0_6b_libero_10_full_finetune.py`

   canonical Wan 配置都是独立入口, `libero_10` 不继承 `libero_object`。训练和
   eval 都使用 `LiberoPromptFromInputs` 在数据 pipeline 中 tokenize prompt,
   产出 `lang_tokens/lang_masks`, 再通过 `vlm_backbone.forward()` 编码成
   WAM head 需要的 context。

   Qwen3VL-0.6B 配置只替换 `vlm_backbone` 和 tokenizer 路径, `video_latent_codec`
   仍然是顶层 Wan2.2 VAE。

   默认从当前 repo 根目录解析 `checkpoints/`、`datasets/` 和
   `work_dirs/`, 也可以用 `FLUXVLA_ROOT=/path/to/FluxVLA` 覆盖。

5. `test/integration_consistency/test_wam_head.py`

   轻量覆盖 registry、batch-level mode collator、backbone-local component
   builder 和 attention mask 语义。

## 删除内容

- 删除 `configs/fastwam/*.py`。
- 删除 `tools/fastwam/*.py`。
- 删除 FastWAM parity / variant / preprocess 相关测试。
- 删除 vendored monolithic wrappers:
  `fastwam.py`, `fastwam_joint.py`, `fastwam_idm.py`。
- 删除公开 registry 中的 `FastWAMVLA`, `FastWAMHead`,
  `FastWAMJointHead`, `FastWAMIDMHead`。

底层的 `ActionDiT`, `MoT`, `WanVideoDiT`, `WanVideoVAE` 和 scheduler
仍保留在 third_party, 因为 WAM 训练和推理直接依赖这些模块。
Wan2.2 checkpoint 装配逻辑移动到 FluxVLA 的 `wan22_loader.py`, 只保留
WAM 当前配置直接用到的四个组件 builder。`WanTextEncoder` 由
`Wan22TextBackbone` build, `Wan22VAE` 由 `WAMVLA` build,
`WanVideoDiT` / `ActionDiT` 由 `WAMHead` build。不再放在 third_party 里,
也不再保留
DiffSynth 风格的通用下载、hash registry、unused state-dict converter、
FastWAM logging helper 或整包 `Wan2.2` loader。
模型 checkpoint 保存/恢复走 FluxVLA runner 的统一格式, 不再保留
FastWAM 特化的 `{mot, proprio_encoder}` checkpoint I/O。

## 和旧实现的关系

早先那版实现更接近把 FastWAM monolith 搬进来, 再通过
`variant` 选择 `fastwam / joint / idm / policy_forward_idm`。这版不保留
那层兼容 API, 只保留 WAM baseline 需要的随机训练目标。

`feat/hqr/support-fastwam` 提供了重要的 split 架构参考, 但本分支已经把
对外入口收口到 WAM, 不再保留 hqr 那版完整 FastWAM variant
配置和 parity 测试。

## 验证入口

```bash
/root/miniconda3/envs/fluxvla/bin/python -m pytest \
  test/integration_consistency/test_wam_head.py

/root/miniconda3/envs/fluxvla/bin/python -m py_compile \
  fluxvla/models/heads/wam_head.py \
  fluxvla/models/vlas/wam_vla.py \
  configs/wam/wam_libero_object_full_finetune.py \
  configs/wam/wam_libero_10_full_finetune.py
```
