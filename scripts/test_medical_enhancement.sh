#!/bin/bash

# 医学图像增强测试脚本
# 用法: bash ./scripts/test_medical_enhancement.sh [dataset_name] [model_name] [gpu_ids]

# 默认参数
DATASET_NAME=${1:-"medical_ct_enhance"}
MODEL_NAME=${2:-"medical_ct_enhance_cyclegan"}
GPU_IDS=${3:-"0"}

echo "开始测试医学图像增强模型..."
echo "数据集: $DATASET_NAME"
echo "模型: $MODEL_NAME"
echo "GPU: $GPU_IDS"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# 测试命令
python test.py \
    --dataroot ./datasets/$DATASET_NAME \
    --name $MODEL_NAME \
    --model medical_cyclegan \
    --dataset_mode medical \
    --direction AtoB \
    --gpu_ids $GPU_IDS \
    --batch_size 1 \
    --load_size 512 \
    --crop_size 256 \
    --preprocess none \
    --no_flip \
    --netG resnet_9blocks \
    --netD basic \
    --norm instance \
    --init_type normal \
    --init_gain 0.02 \
    --no_dropout \
    --epoch latest \
    --num_test 1000 \
    --results_dir ./results/${MODEL_NAME}_test \
    --aspect_ratio 1.0 \
    --medical_modality CT \
    --window_width 400 \
    --window_level 40 \
    --save_metrics \
    --calculate_quality_scores

echo "测试完成！"
echo "结果保存在: ./results/${MODEL_NAME}_test"