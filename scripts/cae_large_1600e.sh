tmp_my_name=${0##*/}
my_name=${tmp_my_name%.*}

OUTPUT_DIR='./output/'$my_name
DATA_PATH=/path/to/imagenet1k/train
TOKENIZER_PATH=./tokenizer-weights

ADDRESS=ADDR_FOR_THIS_MACHINE                                                                                 
NNODES=4     
RANK=RANK_FOR_THIS_MACHINE                                                                                                                        


# ============================ pretraining ============================
OMP_NUM_THREADS=1 python -m torch.distributed.launch \
  --nproc_per_node=8 \
  --nnodes=$NNODES \
  --node_rank=$RANK \
  --master_addr=$ADDRESS \
  --master_port=8899 \
  tools/run_pretraining.py \
  --data_path ${DATA_PATH} \
  --output_dir ${OUTPUT_DIR} \
  --model cae_large_patch16_224_8k_vocab --discrete_vae_weight_path ${TOKENIZER_PATH} \
  --batch_size 64 --lr 1.5e-3 --warmup_epochs 40 --epochs 1600 \
  --clip_grad 3.0 --layer_scale_init_value 1e-5 \
  --imagenet_default_mean_and_std \
  --color_jitter 0 \
  --drop_path 0.1 \
  --sincos_pos_emb \
  --mask_generator block \
  --num_mask_patches 98 \
  --decoder_layer_scale_init_value 1e-5 \
  --no_auto_resume \
  --save_ckpt_freq 100 \
  --exp_name $my_name \
  --regressor_depth 4 \
  --decoder_depth 4 \
  --align_loss_weight 2
  --decoder_embed_dim 1024 \
  --decoder_num_heads 16 \
  --fix_init_weight

# ============================ linear probing ============================
DATA_PATH=/path/to/imagenet1k/
MODEL_PATH=/path/to/pretrained/model

OMP_NUM_THREADS=1 python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=$NNODES \
    --node_rank=$RANK \
    --master_addr=$ADDRESS \
    --master_port=8899 \
    tools/run_linear.py \
    --model cae_large_patch16_224 --data_path $DATA_PATH \
    --finetune $MODEL_PATH \
    --nb_classes 1000 \
    --batch_size 512 \
    --epochs 90 \
    --blr 0.1 \
    --weight_decay 0.0 \
    --dist_eval --data_path ${DATA_PATH} \
    --output_dir $OUTPUT_DIR \
    --log_dir $OUTPUT_DIR \
    --enable_linear_eval \
    --use_cls \
    --dist_eval \
    --save_freq 90 \
    --disable_rel_pos_bias \
    --linear_type standard \
    --exp_name $my_name


# ============================ attentive probing ============================
DATA_PATH=/path/to/imagenet1k/
MODEL_PATH=/path/to/pretrained/model

OMP_NUM_THREADS=1 python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=$NNODES \
    --node_rank=$RANK \
    --master_addr=$ADDRESS \
    --master_port=8899 \
    tools/run_attentive.py \
    --model cae_large_patch16_224 --data_path $DATA_PATH \
    --finetune $MODEL_PATH \
    --nb_classes 1000 --data_set IMNET --imagenet_default_mean_and_std \
    --output_dir $OUTPUT_DIR --batch_size 256 --lr 0.4 --update_freq 1 \
    --warmup_epochs 10 --epochs 90 \
    --weight_decay 0 --smoothing 0.0 --layer_decay 1.0 --drop_path 0.0 \
    --color_jitter 0.0 --mixup 0.0 --cutmix 0.0 --reprob 0.0 \
    --opt sgd --momentum 0.9 \
    --enable_linear_eval \
    --use_cls \
    --dist_eval \
    --no_auto_resume \
    --save_ckpt_freq 50 \
    --linear_type attentive \
    --exp_name $my_name
