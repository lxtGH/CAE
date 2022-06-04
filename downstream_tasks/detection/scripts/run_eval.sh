#!/usr/bin/env bash

echo "EVAL MODEL:"$MODEL
python -m torch.distributed.launch --nproc_per_node=8 \
    evaluation/object_detection/test.py \
    $CONFIG \
    $MODEL \
    --launcher pytorch \
    --eval bbox segm \
    --cfg-options model.backbone.use_checkpoint=True \
    ${@:6}

