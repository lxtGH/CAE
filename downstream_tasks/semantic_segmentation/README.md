# ADE20k Semantic segmentation with CAE

## Getting started 

1. Install the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) library and some required packages.

```bash
pip install mmcv-full==1.3.0 mmsegmentation==0.11.0
pip install scipy timm==0.3.2
```

2. Install [apex](https://github.com/NVIDIA/apex) for mixed-precision training

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

3. Follow the guide in [mmseg](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/dataset_prepare.md) to prepare the ADE20k dataset.


## Fine-tuning

Command format:
```
tools/dist_train.sh <CONFIG_PATH> <NUM_GPUS>  --work-dir <SAVE_PATH> --seed 0  --deterministic --options model.pretrained=<PRETRAIN_CHECKPOINT_PATH>
```

For example, using a CAE-base backbone with UperNet:
```bash
bash tools/dist_train.sh \
    configs_local/cae/upernet/upernet_cae_base_12_512_slide_160k_ade20k_pt_4e-4.py 8 \
    --work-dir /path/to/save --seed 0  --deterministic \
    --options model.pretrained=<PRETRAIN_CHECKPOINT_PATH>
```

More config files can be found at [`configs_local/cae/upernet`](configs_local/cae/upernet).


## Evaluation

Command format:
```
tools/dist_test.sh  <CONFIG_PATH> <CHECKPOINT_PATH> <NUM_GPUS> --eval mIoU
```

For example, evaluate a CAE-base backbone with UperNet:

```bash
bash tools/dist_test.sh configs_local/cae/upernet/upernet_cae_base_12_512_slide_160k_ade20k_pt_4e-4.py \ 
    <CHECKPOINT_PATH> 8 --eval mIoU
```

Please note that, the evaluation will be automatically conducted during training.

## Results (pretrined models are trained on ImageNet-1K without label)

| Backbone | #Pretrained Epoch | mIoU | Config                                   |
| -------- | ----------------- | ---- | ---------------------------------------- |
| ViT-B    | 300               | 48.1 | [3e-4](./configs_local/cae/upernet/upernet_cae_base_12_512_slide_160k_ade20k_pt_3e-4.py) |
| ViT-B    | 800               | 49.7 | [2e-4](./configs_local/cae/upernet/upernet_cae_base_12_512_slide_160k_ade20k_pt_2e-4.py) |
| ViT-B    | 1600              | 50.3 | [1e-4](./configs_local/cae/upernet/upernet_cae_base_12_512_slide_160k_ade20k_pt_1e-4.py) |
| ViT-L    | 1600              | 54.9 | [4e-5](./configs_local/cae/upernet/upernet_cae_large_24_512_slide_160k_ade20k_pt_decay095_4e-5_dp015.py) |

We find that, if the pretrained model is better, a smaller learning rate is more suitable. However, different learning rates will not lead to significantly different results. For example, 800-epoch pretrained ViT-B could obtain 49.6 mIoU (averaged from two runs) with lr=4e-4.

## Acknowledgment 

This code is built using the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) library, [Timm](https://github.com/rwightman/pytorch-image-models) library, the [Swin](https://github.com/microsoft/Swin-Transformer) repository, [XCiT](https://github.com/facebookresearch/xcit) and the [SETR](https://github.com/fudan-zvg/SETR) repository.
