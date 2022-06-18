
# COCO Detection and Instance segmentation with CAE

# Installation

Please install [PyTorch](https://pytorch.org/). This codebase has been developed with python version 3.6, PyTorch version 1.7.1, CUDA 11.0 and torchvision 0.8.2. To get the full dependencies, please run:

```bash
pip3 install -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.1/index.html mmcv-full==1.3.9
pip3 install pytest-runner scipy tensorboardX faiss-gpu==1.6.1 tqdm lmdb sklearn pyarrow==2.0.0 timm DALL-E munkres six einops

# install apex
pip3 install git+https://github.com/NVIDIA/apex \
    --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext"

# install mmdetection for object detection & instance segmentation
git clone https://github.com/SwinTransformer/Swin-Transformer-Object-Detection
cd Swin-Transformer-Object-Detection
pip3 install -r requirements/build.txt
pip3 install -v -e .
cd ..
```


## Fine-tuning with Mask R-CNN
#### We use 16 GPUs for these experiments, $NNODES = 2.

- To train ViT-B/16 with Mask R-CNN as the task layer, run:
```bash
python -m torch.distributed.launch --nproc_per_node=8 \
    --nnodes=$NNODES \
    --node_rank=$RANK \
    --master_addr=$ADDRESS \
    --master_port=$PORT \
    evaluation/object_detection/train.py evaluation/object_detection/configs/mask_rcnn/vit_base_giou_4conv1f_coco_maskrcnn_1x_cae_sincos_init0.1_lr00003.py \
    --launcher pytorch \
    --work-dir $OUTPUT_DIR \
    --no-validate \
    --deterministic \
    --cfg-options model.backbone.use_checkpoint=True \
    model.pretrained=$PRETRAINED \
    ${@:6}
```

- To train ViT-L/16 with Mask R-CNN as the task layer, run:
```bash
python -m torch.distributed.launch --nproc_per_node=8 \
    --nnodes=$NNODES \
    --node_rank=$RANK \
    --master_addr=$ADDRESS \
    --master_port=$PORT \
    evaluation/object_detection/train.py evaluation/object_detection/configs/mask_rcnn/vit_large_giou_4conv1f_coco_maskrcnn_1x_cae_sincos_init0.1_lr00002_lrdr0.85_dp0.2.py \
    --launcher pytorch \
    --work-dir $OUTPUT_DIR \
    --no-validate \
    --deterministic \
    --cfg-options model.backbone.use_checkpoint=True \
	model.pretrained=$PRETRAINED \
    ${@:6}
```

- To evaluate Mask R-CNN, run:
```bash
python -m torch.distributed.launch --nproc_per_node=8 \
    evaluation/object_detection/test.py \
    $CONFIG \
    $MODEL \
    --launcher pytorch \
    --eval bbox segm \
    --cfg-options model.backbone.use_checkpoint=True \
    ${@:6}
```

## Results (pretrined models are trained on ImageNet-1K without label)
| Backbone | #Pretrained Epoch | Object Det | Instance Seg |
| -------- | ----------------- | ---------- | ------------ |
| ViT-B    | 300               | 48.3       | 42.7         |
| ViT-B    | 800               | 49.9       | 43.9         |
| ViT-B    | 1600              | 50.3       | 44.2         |
| ViT-L    | 1600              | 54.5       | 47.5         |


## Acknowledgement

This repository is built using the [IBOT repository](https://github.com/bytedance/ibot). Thanks for their open-source code!
