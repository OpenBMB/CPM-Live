#! /bin/bash
export CUDA_VISIBLE_DEVICES=0,1
GPUS_PER_NODE=2

NNODES=1
MASTER_ADDR=localhost
MASTER_PORT=12345

OPTS=""
OPTS+=" --dataset-name Math23K"
OPTS+=" --dataset-path path/to/Math23K"
OPTS+=" --output-path Math23K/output/path"
OPTS+=" --model-path path/to/cpm-ant-10b.pt"
OPTS+=" --config-path path/to/cpm-ant-10b.json"
OPTS+=" --batch-size 16"
OPTS+=" --early-stop-patience 10"
OPTS+=" --eval-interval 50"
OPTS+=" --tune-maxlen 256"
OPTS+=" --lr 5e-3"
OPTS+=" --warmup-iters 50"
OPTS+=" --epochs 100"
OPTS+=" --infer-maxlen 32"

TUNE_CMD="torchrun --nnodes=${NNODES} --nproc_per_node=${GPUS_PER_NODE} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} tune_cpm_ant.py ${OPTS}"

echo ${TUNE_CMD}
$TUNE_CMD

INFER_CMD="python -u infer_cpm_ant.py ${OPTS}"
echo ${INFER_CMD}
$INFER_CMD
