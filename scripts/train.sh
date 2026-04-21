#!/bin/bash

MLP_WORKER_GPU=8
MLP_WORKER_NUM=1
MLP_ROLE_INDEX=0
MLP_WORKER_0_HOST=localhost
MLP_WORKER_0_PORT=29500

CONFIG=${1:-"configs/gr00t/gr00t_hud04_rtc_full_finetune.py"}
WORK_DIR=${2:-"/limx_embmc/tos/users/jace/fluxvla/gr00t_rtc_0413_xyyaw_delta_wrist_camera"}

# export CUDA_VISIBLE_DEVICES=0,1

torchrun \
  --nproc-per-node="${MLP_WORKER_GPU}" \
  --nnodes="${MLP_WORKER_NUM}" \
  --node_rank="${MLP_ROLE_INDEX}" \
  --master_addr="${MLP_WORKER_0_HOST}" \
  --master_port="${MLP_WORKER_0_PORT}" \
  "scripts/train.py" \
  --config "${CONFIG}" \
  --work-dir "${WORK_DIR}" \
  ${@:3}
