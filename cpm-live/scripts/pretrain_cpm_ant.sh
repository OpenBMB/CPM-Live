#! /bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
GPUS_PER_NODE=8

NNODES=1
MASTER_ADDR=$(tail -n 1 /etc/hosts | cut -d: -f2 | awk '{ print $2}')
MASTER_PORT=12345
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $2 \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

OPTS=""
OPTS+=" --model-config config/cpm-ant-10b.json"
OPTS+=" --batch-size 64"
OPTS+=" --train-iters 200000"
OPTS+=" --save-iters 500"
OPTS+=" --save-name cpm_live_checkpoint"
OPTS+=" --max-length 512"
OPTS+=" --save results/"
OPTS+=" --lr 0.1"
OPTS+=" --inspect-iters 100"
OPTS+=" --warmup-iters 2000"
OPTS+=" --lr-decay-style noam"
OPTS+=" --weight-decay 0.01"
OPTS+=" --clip-grad 4.0"
OPTS+=" --loss-scale 1048576"
OPTS+=" --start-step 0"
OPTS+=" --log-dir logs/tensorboard/cpm_live_48_4096/"
# OPTS+=" --load results/cpm_live_checkpoint.pt"

CMD="torchrun --nnodes=${NNODES} --nproc_per_node=${GPUS_PER_NODE} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} pretrain_cpm_ant.py ${OPTS}"

echo ${CMD}
$CMD

