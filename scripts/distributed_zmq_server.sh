#!/bin/sh
# Launch one server supervisor per cloud node. The supervisor creates one
# model worker per visible GPU; torchrun itself starts one rank per node.
set -eu

REPO_ROOT=$(CDPATH= cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"

NNODES=${WORLD_SIZE:-${MLP_WORKER_NUM:-1}}
NODE_RANK=${RANK:-${MLP_ROLE_INDEX:-0}}
RENDEZVOUS_HOST=${MASTER_ADDR:-${MLP_WORKER_0_HOST:-localhost}}
RENDEZVOUS_PORT=${MASTER_PORT:-${MLP_WORKER_0_PORT:-29500}}
LOCAL_WORKERS=${NPROC_PER_NODE:-${MLP_WORKER_GPU:-}}

if [ -n "$LOCAL_WORKERS" ]; then
    export NPROC_PER_NODE=$LOCAL_WORKERS
fi

exec "${PYTHON:-python}" -m torch.distributed.run \
    --nnodes="$NNODES" \
    --nproc-per-node=1 \
    --node-rank="$NODE_RANK" \
    --master-addr="$RENDEZVOUS_HOST" \
    --master-port="$RENDEZVOUS_PORT" \
    scripts/distributed_zmq_server.py \
    "$@"
