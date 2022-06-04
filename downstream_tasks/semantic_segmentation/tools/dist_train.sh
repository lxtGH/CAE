#!/usr/bin/env bash
CONFIG=$1
GPUS=$2
PORT=7956

OMP_NUM_THREADS=1 python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/train.py $CONFIG --launcher pytorch ${@:3} \
