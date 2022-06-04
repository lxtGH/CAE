tmp_my_name=${0##*/}
my_name=${tmp_my_name%.*}

OUTPUT_DIR='./output/'$my_name
DATA_PATH=/path/to/imagenet1k/train
TOKENIZER_PATH=./tokenizer-weights

ADDRESS=ADDR_FOR_THIS_MACHINE                                                                                 
NNODES=4     
RANK=RANK_FOR_THIS_MACHINE                                                                                                                        

MODEL_PATH=/path/to/pretrained/model

OMP_NUM_THREADS=1 python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=$NNODES \
    --node_rank=$RANK \
    --master_addr=$ADDRESS \
    --master_port=8899 \
    tools/run_class_finetuning.py \
    --model cae_base_patch16_224  --data_path $DATA_PATH \
    --finetune $MODEL_PATH \
    --nb_classes 1000 --data_set IMNET \
    --output_dir $OUTPUT_DIR \
    --batch_size 128 \
    --lr 8e-3 --update_freq 1 \
    --warmup_epochs 5 --epochs 100 --layer_decay 0.65 --drop_path 0.1 \
    --weight_decay 0.05 --mixup 0.8 --cutmix 1.0 \
	--sin_pos_emb \
    --dist_eval \
    --no_auto_resume \
    --exp_name $my_name \
    --imagenet_default_mean_and_std




