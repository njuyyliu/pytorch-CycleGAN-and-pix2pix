#!/bin/bash

# 医学图像增强CycleGAN训练脚本
# 用法: bash ./scripts/train_medical_enhancement.sh [dataset_name] [gpu_ids]

# 默认参数
DATASET_NAME=${1:-"medical_ct_enhance"}
GPU_IDS=${2:-"0,1"}

echo "开始训练医学图像增强模型..."
echo "数据集: $DATASET_NAME"
echo "GPU: $GPU_IDS"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# 创建必要的目录
mkdir -p ./datasets/$DATASET_NAME
mkdir -p ./checkpoints/${DATASET_NAME}_cyclegan
mkdir -p ./results/${DATASET_NAME}_cyclegan

# 训练命令
python train.py \
    --dataroot ./datasets/$DATASET_NAME \
    --name ${DATASET_NAME}_cyclegan \
    --model medical_cyclegan \
    --dataset_mode medical \
    --direction AtoB \
    --gpu_ids $GPU_IDS \
    --batch_size 4 \
    --load_size 512 \
    --crop_size 256 \
    --preprocess resize_and_crop \
    --no_flip \
    --netG resnet_9blocks \
    --netD basic \
    --norm instance \
    --init_type normal \
    --init_gain 0.02 \
    --n_epochs 100 \
    --n_epochs_decay 50 \
    --lr 0.0002 \
    --lr_policy linear \
    --beta1 0.5 \
    --lambda_A 10.0 \
    --lambda_B 10.0 \
    --lambda_identity 0.5 \
    --lambda_medical 1.0 \
    --use_medical_loss \
    --preserve_structure \
    --enhance_contrast \
    --pool_size 50 \
    --no_dropout \
    --display_freq 100 \
    --print_freq 100 \
    --save_latest_freq 5000 \
    --save_epoch_freq 10 \
    --update_html_freq 1000 \
    --display_winsize 256 \
    --max_dataset_size 10000 \
    --medical_modality CT \
    --window_width 400 \
    --window_level 40 \
    --use_wandb \
    --wandb_project medical_image_enhancement

echo "训练完成！"