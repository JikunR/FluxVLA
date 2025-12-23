# fluxvla

## Installation

### Install pytorch

```
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

### Install LIBERO

```
git clone https://github.com/Lifelong-Robot-Learning/LIBERO
cd LIBERO
pip install -r requirements.txt
pip install -e .
```

Due to PyTorch version changes, LIBERO may require some accommodations; in particular, the way we use torch.load might need to be updated.

### Install transformers

```
pip install transformers==4.53.2
```

### Install flash-attention

```
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout v2.5.5
MAX_JOBS=4 python setup.py install
```

### Install dlimp

```
git clone https://github.com/kvablack/dlimp
cd dlimp
pip install -e .
```

### Install fluxvla

```
pip install -r requirements.txt
python setup.py develop
```

To support the evaluation of Libero on devices without ray tracing (such as the A100, etc.), please follow the guidelines outlined in the [Eval on A100](https://cwjgfm21di.feishu.cn/wiki/GGRUwx978isixUkNFQccidZWnJe?from=from_copylink) specification.

## Data Preparation

To train models using fluxvla on private datasets, organize the datasets in the following format.

```
├── data
│   └── chunk000
│   │   └── episode_000000.parquet
│   │   └── episode_000001.parquet
│   │   └── ... (more parquets)
│   │   └── episode_00000N.parquet
│   └── chunk001
│   └── ... (more chunks)
│   └── chunk00N
├── meta
│   └── episodes.jsonl
│   └── episodes_stats.jsonl
│   └── info.json
│   └── tasks.jsonl
├── videos
│   └── chunk000
│   │   └── camera name 0
│   │   │   └── episode_000000.mp4
│   │   │   └── episode_000001.mp4
│   │   │   └── ...(more mp4s)
│   │   │   └── episode_00000N.mp4
│   │   └── camera name 1
│   └── chunk001
│   └── ... (more chunks)
│   └── chunk00N
```

## Features

- Support OpenVLA, LlavaVLA, Gr00t, Pi0 and Pi0.5.
- Support llama, gemma and qwen llm backbones.
- Support dinosiglip vision backbone.
- Support paligemma and qwenvl vlm backbones.
- Support multi-gpu evaluation.
- Support evaluate libero on devices without ray tracing.
- Support eval-after-train.
- Support both FSDP and DDP, support lora training mode.
- Support Parquet datasets and enable the loading of data in the LeRobot format.
- Support resuming training from checkpoints.

## Usage

### Debug locally

You can use the `train_local.sh` script for convenient local single-machine training, or use `torchrun` directly.

**Using train_local.sh (Recommended):**

```
export WANDB_MODE=disabled
export HF_ENDPOINT="https://hf-mirror.com"
bash scripts/train_local.sh [CONFIG] [WORK_DIR] [NUM_GPUS] [OTHER_ARGS...]
```

For example, if you train PI0.5 on libero dataset with 2 GPUs:

```
export WANDB_MODE=disabled
export HF_ENDPOINT="https://hf-mirror.com"
bash scripts/train_local.sh configs/pi05/pi05_paligemma_libero10_full_finetune.py ./work_dirs/pi05_paligemma_libero10_full_finetune 2 --cfg-options train_dataloader.per_device_batch_size=2
```

**Using torchrun directly:**

```
/root/miniconda3/bin/torchrun --standalone --nnodes 1 --nproc-per-node [NUM_GPUS] scripts/train.py --config [CONFIG_PATH] --work-dir [WORK_DIR] --cfg-options train_dataloader.per_device_batch_size=[PER_DEVICE_BATCH_SIZE]
```

For example:

```
export WANDB_MODE=disabled
export HF_ENDPOINT="https://hf-mirror.com"
/root/miniconda3/bin/torchrun --standalone --nnodes 1 --nproc-per-node 2 scripts/train.py --config configs/pi05/pi05_paligemma_libero10_full_finetune.py --work-dir ./work_dirs/pi05_paligemma_libero10_full_finetune --cfg-options train_dataloader.per_device_batch_size=2
```

### Eval locally

```
/root/miniconda3/bin/torchrun --standalone --nnodes 1 --nproc-per-node [NUM_GPUS] scripts/eval.py --config [CONFIG_PATH] --ckpt-path [CKPT_PATH] --cfg-options [CFG_OPTIONS]
```

### Train on cluster

```
export WANDB_MODE=disabled
export HF_ENDPOINT="https://hf-mirror.com"
bash scripts/train.sh [CONFIG] [WORK_DIR] --cfg-options train_dataloader.per_device_batch_size=[PER_DEVICE_BATCH_SIZE] train_dataloader.batch_size=[GLOBAL_BATCH_SIZE] runner.max_steps=[MAX_STEPS] runner.save_interval=[SAVE_INTERVAL] --eval-after-train
```

### Resume training from checkpoint

To resume training from a checkpoint, use the `--resume-from` parameter to specify the path to the checkpoint file. The training will resume from the saved global step, epoch, model state, and optimizer state.

**Local training example:**

```
export WANDB_MODE=disabled
export HF_ENDPOINT="https://hf-mirror.com"
/root/miniconda3/bin/torchrun --standalone --nnodes 1 --nproc-per-node 2 scripts/train.py \
  --config configs/pi05/pi05_paligemma_libero10_full_finetune.py \
  --work-dir ./work_dirs/pi05_paligemma_libero10_full_finetune \
  --resume-from ./work_dirs/pi05_paligemma_libero10_full_finetune/checkpoints/checkpoint_epoch_5.pth \
  --cfg-options train_dataloader.per_device_batch_size=2
```

**Cluster training example:**

```
export WANDB_MODE=disabled
export HF_ENDPOINT="https://hf-mirror.com"
bash scripts/train.sh [CONFIG] [WORK_DIR] \
  --resume-from [CHECKPOINT_PATH] \
  --cfg-options train_dataloader.per_device_batch_size=[PER_DEVICE_BATCH_SIZE] runner.max_steps=[MAX_STEPS]
```

### Eval on cluster

```
export WANDB_MODE=disabled
export HF_ENDPOINT="https://hf-mirror.com"
bash scripts/eval.sh [CONFIG] [CKPT_PATH] --cfg-options [CFG_OPTIONS]
```

### Inference on real robot

To run inference on the real robot, first install the environment on the robot, then execute the following command.

```
python scripts/inference_real_robot.py --config [CONFIG] -- ckpt-path [CKPT_PATH]
```

## Support

If you encounter any issues while using this repository, don't hesitate to reach out to us. You can contact @mason directly or open an issue on GitLab for assistance.

## Roadmap

- More vision backbones will be supported.
- More vlm backbones will be supported.
- More VLA methods will be supported.
- Training with VLM data or Chain-of-Thought (CoT) data will be supported.
- The RLDS dataset will be deprecated and replaced by a Parquet dataset.
- The logger functionality will be fully implemented.
- issacsim will be supported.

## Finetune huggingface models

We support to finetune models directly downloaded from huggingface. We provide `configs/openvla/openvla_dino_siglip_llama2_libero_10_hf_finetune.py` as an example. Todo that, you should place the Python file containing all the necessary modules to build the structure of the model you intend to train into the `fluxvla/models/hf_models` directory. You can then utilize the `ddp_hf_finetune_runner` to complete the fine-tuning process.
