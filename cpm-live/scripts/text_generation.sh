#! /bin/bash
export CUDA_VISIBLE_DEVICES=0
GPUS_PER_NODE=1
NNODES=1
MASTER_ADDR=localhost
MASTER_PORT=12345

CMD="torchrun --nnodes=${NNODES} --nproc_per_node=${GPUS_PER_NODE} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} text_generation.py"

echo ${CMD}
$CMD

