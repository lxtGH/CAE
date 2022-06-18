#!/usr/bin/env bash

python -m torch.distributed.launch --nproc_per_node=8 \
    --nnodes=$NNODES \
    --node_rank=$RANK \
    --master_addr=$ADDRESS \
    --master_port=$PORT \
    evaluation/object_detection/train.py \
    evaluation/object_detection/configs/mask_rcnn/vit_large_giou_4conv1f_coco_maskrcnn_1x_cae_sincos_init0.1_lr00002_lrdr0.85_dp0.2.py \
    --launcher pytorch \
    --work-dir $OUTPUT_DIR \
    --no-validate \
    --deterministic \
    --cfg-options model.backbone.use_checkpoint=True \
	model.pretrained=$PRETRAINED \
    ${@:6}

