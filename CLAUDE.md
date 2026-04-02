# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FluxVLA is a unified, modular VLA (Vision-Language-Action) codebase for robot manipulation. It supports multiple VLA architectures (OpenVLA, LlavaVLA, GR00T, PI0, PI0.5) with pluggable vision backbones, LLM backbones, VLM backbones, training strategies (FSDP/DDP/LoRA), and inference acceleration (Triton fused kernels, CUDA Graphs, CUDA custom operators).

## Build & Install

```bash
conda create -n fluxvla python=3.10 -y && conda activate fluxvla
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
MAX_JOBS=8 pip install flash-attn==2.5.5 --no-build-isolation
conda install -c conda-forge av=14.4.0
pip install -r requirements.txt
pip install --no-build-isolation -e .   # builds CUDA extensions (rotary embeddings, matmul_bias)
```

The `pip install -e .` step compiles three CUDA C++ extensions defined in `setup.py`: `gemma_rotary_embedding_ext`, `rotary_pos_embedding_ext`, and `matmul_bias_ext`.

## Common Commands

### Training (local)
```bash
export WANDB_MODE=disabled
torchrun --standalone --nnodes 1 --nproc-per-node NUM_GPUS scripts/train.py \
  --config CONFIG_PATH --work-dir WORK_DIR \
  --cfg-options train_dataloader.per_device_batch_size=BS
```

### Evaluation (local)
```bash
torchrun --standalone --nnodes 1 --nproc-per-node NUM_GPUS scripts/eval.py \
  --config CONFIG_PATH --ckpt-path CKPT_PATH
```

### Cluster training/eval
```bash
bash scripts/train.sh CONFIG WORK_DIR [--cfg-options ...]
bash scripts/eval.sh CONFIG CKPT_PATH [--cfg-options ...]
```

### Real robot inference
```bash
python scripts/inference_real_robot.py --config CONFIG --ckpt-path CKPT_PATH
```

### RTC testing
```bash
python scripts/test_rtc.py --config CONFIG --checkpoint CKPT --prefix_len 5 --output_dir work_dirs/rtc_test
```

### Running a single test
```bash
pytest test/test_models/test_vla.py -k "TestPI05" -v
```
Tests require model checkpoints in `./checkpoints/` and are skipped when checkpoints are absent.

## Code Style

Pre-commit hooks enforce: flake8, isort, yapf, codespell, trailing-whitespace, double-quote-string-fixer, clang-format + cpplint (C++/CUDA). The `fluxvla/models/third_party_models/eagle2_hg_model/` directory is excluded from all Python linting hooks.

Key style rules:
- **yapf** formatting (PEP8-based), **isort** for imports (line_length=79, multi_line_output=0)
- Use single quotes for strings (double-quote-string-fixer is active)
- C++/CUDA files: clang-format with `.clang-format` config, cpplint for linting

## Architecture

### Registry-Driven Component System

All components are registered via a custom `Registry` (based on mmengine) in `fluxvla/engines/utils/root.py`. Registries include: `VLAS`, `LLM_BACKBONES`, `VISION_BACKBONES`, `VLM_BACKBONES`, `PROJECTORS`, `HEADS`, `RUNNERS`, `DATASETS`, `COLLATORS`, `TRANSFORMS`, `TOKENIZERS`, `METRICS`, `PROCESSORS`, `OPERATORS`.

Components are instantiated from config dicts with a `type` key via `build_*_from_cfg()` functions in `fluxvla/engines/utils/builder.py`. Configuration files (Python dicts in `configs/`) are loaded with `mmengine.Config.fromfile()` and can be overridden at CLI via `--cfg-options key=value`.

### Config Structure

Each config file (`configs/{model_family}/*.py`) defines three top-level dicts:
- `model`: VLA architecture with nested backbone, projector, and head configs
- `train_dataloader`: dataset type, transforms pipeline, batch settings
- `runner`: training runner type (FSDPTrainRunner/DDPTrainRunner), optimizer, scheduler, collator, metric
- `eval` (optional): evaluation runner config (e.g. LiberoEvalRunner)

### VLA Models (`fluxvla/models/vlas/`)

| Class | Architecture | Action Head |
|-------|-------------|-------------|
| `OpenVLA` | DinoSigLIP + LLaMA2 | OpenVLAHead (token-based) |
| `LlavaVLA` | Vision + LLM + FlowMatchingHead | LlavaActionHead / FlowMatchingHead |
| `PI0FlowMatching` | PaliGemma + Gemma expert | Flow matching (denoising on model) |
| `PI05FlowMatching` | PaliGemma + Gemma expert | Flow matching with action chunking |
| `PI05FlowMatchingInference` | Full CUDA Graph-accelerated PI0.5 | Fused Triton ops |

### Model Composition

VLAs are composed of pluggable sub-modules, each from its own registry:
- **Vision backbones** (`VISION_BACKBONES`): DinoSigLIPViT, SigLIPViT, SigLIPViTInference
- **LLM backbones** (`LLM_BACKBONES`): Gemma, ConditionGemma, LLaMA2, Qwen2, HFCausalLLM
- **VLM backbones** (`VLM_BACKBONES`): Eagle, PaliGemma, QWen2.5-VL
- **Projectors** (`PROJECTORS`): Linear, MLP, FusedMLP
- **Action heads** (`HEADS`): FlowMatching, FlowMatchingInference, LlavaAction, OpenVLA

### Training Runners (`fluxvla/engines/runners/`)

- `FSDPTrainRunner`: Fully Sharded Data Parallel training
- `DDPTrainRunner`: Distributed Data Parallel training
- Both inherit from `BaseTrainRunner`

### Inference/Eval Runners

- `LiberoEvalRunner`: LIBERO benchmark sim evaluation (multi-GPU)
- `LiberoInferenceRunner`, `AlohaInferenceRunner`, `URInferenceRunner`: robot-specific inference
- `AlohaRTCInferenceRunner`: Aloha with RTC (Real-Time Chunking)

### Data Pipeline

- **Datasets**: `ParquetDataset` (LeRobot v2 format), `RLDSDataset` (legacy), `DistributedRepeatingDataset` (wrapper)
- **Transforms** (`fluxvla/transforms/`): composable pipeline stages — input processing, image resize/normalize, prompt generation, state/action normalization, tokenization
- **Collators**: `DictCollator`, `NestedCollator`, `PaddedCollatorForActionPrediction`, `PaddedCollatorForLanguageModeling`

### Inference Acceleration (`fluxvla/ops/`)

- `fluxvla/ops/triton/`: Fused Triton kernels (norm+matmul, QKV+RoPE, gated MLP, position embedding)
- `fluxvla/ops/cuda/`: CUDA C++ extensions (gemma_rotary_embedding, rotary_pos_embedding, matmul_bias via cublasLt)

### Key Directories

- `configs/`: model/training/eval configurations organized by model family (gr00t, pi0, pi05, llava, openvla)
- `checkpoints/`: pretrained model weights (gitignored, downloaded separately)
- `datasets/`: training data in LeRobot Parquet format (gitignored)
- `work_dirs/`: training outputs, logs, saved checkpoints
- `docs/`: RTC and inference acceleration documentation
- `test/`: pytest tests organized by component (test_models, test_ops, test_tokenizers, test_transforms)

## Important Patterns

- Checkpoint loading uses `name_mapping` dicts in model configs to remap pretrained weight keys to FluxVLA's module hierarchy
- The `--cfg-options` CLI flag (via mmengine's `DictAction`) allows dot-separated key override of any config value
- `--eval-after-train` flag on `scripts/train.py` chains evaluation after training completes
- `--resume-from` restores full training state (model, optimizer, step, epoch)
- wandb configuration is entirely via environment variables (`WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_MODE`)
