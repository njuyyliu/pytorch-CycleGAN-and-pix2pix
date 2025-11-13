# 医学图像增强使用指南

本指南介绍如何使用CycleGAN进行医学图像质量增强。

## 目录
1. [环境准备](#环境准备)
2. [数据准备](#数据准备)
3. [模型训练](#模型训练)
4. [模型测试](#模型测试)
5. [结果评估](#结果评估)
6. [参数调优](#参数调优)
7. [常见问题](#常见问题)

## 环境准备

### 安装依赖
```bash
# 安装PyTorch和相关依赖
conda create -n medical_enhancement python=3.11
conda activate medical_enhancement
pip install torch torchvision torchaudio

# 安装医学图像处理库
pip install pydicom SimpleITK opencv-python
pip install scikit-image matplotlib pandas seaborn
pip install wandb  # 可选，用于实验追踪

# 安装项目依赖
pip install -r requirements.txt
```

### 依赖说明
- `pydicom`: DICOM文件处理
- `SimpleITK`: 医学图像格式转换
- `scikit-image`: 图像质量评估
- `wandb`: 实验追踪和可视化

## 数据准备

### 数据集结构
```
datasets/medical_ct_enhance/
├── trainA/          # 低质量CT图像
│   ├── patient001_001.dcm
│   ├── patient002_001.dcm
│   └── ...
├── trainB/          # 高质量CT图像
│   ├── patient001_hq_001.dcm
│   ├── patient002_hq_001.dcm
│   └── ...
├── testA/           # 测试用低质量图像
└── testB/           # 测试用高质量图像
```

### 支持的图像格式
- DICOM格式 (.dcm)
- 标准图像格式 (.png, .jpg, .tiff)

### 数据预处理
1. **DICOM图像处理**：
   - 自动应用窗宽窗位
   - 归一化到[0,255]范围
   - 支持不同CT/MRI协议

2. **数据增强**（仅训练时）：
   - 轻微旋转（±5°）
   - 对比度调整（0.8-1.2倍）
   - 高斯噪声注入

## 模型训练

### 基础训练
```bash
# 使用默认参数训练
bash ./scripts/train_medical_enhancement.sh

# 指定数据集和GPU
bash ./scripts/train_medical_enhancement.sh medical_ct_enhance 0,1,2,3
```

### 自定义训练参数
```bash
python train.py \
    --dataroot ./datasets/your_dataset \
    --name your_model_name \
    --model medical_cyclegan \
    --dataset_mode medical \
    --medical_modality CT \
    --window_width 400 \
    --window_level 40 \
    --use_medical_loss \
    --preserve_structure \
    --enhance_contrast \
    --n_epochs 200 \
    --n_epochs_decay 100 \
    --batch_size 8 \
    --lr 0.0001
```

### 关键参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--medical_modality` | 医学成像模态 | CT/MRI/XRay |
| `--window_width` | CT窗宽 | 根据具体需求调整 |
| `--window_level` | CT窗位 | 根据具体需求调整 |
| `--use_medical_loss` | 使用医学专用损失 | True |
| `--preserve_structure` | 保持结构信息 | True |
| `--enhance_contrast` | 增强对比度 | True |
| `--lambda_medical` | 医学损失权重 | 1.0-2.0 |

## 模型测试

### 基础测试
```bash
# 使用训练好的模型测试
bash ./scripts/test_medical_enhancement.sh medical_ct_enhance medical_ct_enhance_cyclegan 0
```

### 批量处理
```python
import torch
from models.medical_cyclegan_model import MedicalCycleGANModel
from data.medical_dataset import MedicalEnhancementDataset

# 加载模型
model = MedicalCycleGANModel()
model.initialize(opt)
model.setup(opt)
model.eval()

# 批量处理图像
dataset = MedicalEnhancementDataset(opt)
for i, data in enumerate(dataset):
    model.set_input(data)
    model.test()
    # 保存结果
```

## 结果评估

### 自动评估
```bash
# 使用评估工具
python -c "
from util.medical_evaluator import MedicalImageEvaluator
evaluator = MedicalImageEvaluator()
avg_metrics, std_metrics = evaluator.evaluate_dataset(
    'results/model_test/fake_B',
    'datasets/testB',
    'evaluation_results'
)
"
```

### 评估指标说明

1. **图像质量指标**：
   - PSNR：峰值信噪比（越高越好）
   - SSIM：结构相似性（0-1，越高越好）

2. **医学特定指标**：
   - 对比度改善：对比度提升比例
   - 噪声抑制：噪声降低比例
   - 边缘保持：结构信息保持度
   - 信息熵：图像信息量

3. **可视化输出**：
   - 增强效果对比图
   - 质量指标可视化
   - 详细评估报告

## 参数调优指南

### 针对不同医学模态的优化

#### CT图像增强
```bash
python train.py \
    --medical_modality CT \
    --window_width 400 \
    --window_level 40 \
    --load_size 512 \
    --crop_size 256 \
    --batch_size 4 \
    --lambda_medical 1.5
```

#### MRI图像增强
```bash
python train.py \
    --medical_modality MRI \
    --load_size 256 \
    --crop_size 256 \
    --batch_size 8 \
    --lambda_medical 2.0 \
    --preserve_structure
```

#### X光图像增强
```bash
python train.py \
    --medical_modality XRay \
    --load_size 512 \
    --crop_size 512 \
    --batch_size 2 \
    --lambda_medical 1.0 \
    --enhance_contrast
```

### 训练策略优化

1. **渐进式训练**：
   ```bash
   # 阶段1：基础特征学习（50 epochs）
   --n_epochs 50 --lr 0.0002 --lambda_medical 0.5

   # 阶段2：细节增强（100 epochs）
   --n_epochs 100 --lr 0.0001 --lambda_medical 1.5

   # 阶段3：质量优化（50 epochs）
   --n_epochs 50 --lr 0.00005 --lambda_medical 2.0
   ```

2. **数据集平衡**：
   - 确保低质量和高质量图像数量平衡
   - 考虑不同扫描仪和协议的数据分布

3. **监控指标**：
   - 关注SSIM和PSNR的变化趋势
   - 监控医学特定指标的改善
   - 定期检查生成图像的质量

## 常见问题

### Q: 如何处理DICOM文件？
A: 项目自动支持DICOM文件处理。确保在训练时添加`--use_dicom`参数。

### Q: 训练过程不收敛怎么办？
A:
1. 检查数据质量和预处理
2. 调整学习率和损失权重
3. 增加训练轮数
4. 检查GPU内存是否充足

### Q: 生成图像出现伪影怎么办？
A:
1. 减少dropout使用
2. 调整医学损失权重
3. 检查原始数据质量
4. 考虑增加训练数据

### Q: 如何评估医学图像增强效果？
A: 使用提供的`MedicalImageEvaluator`工具，它会计算PSNR、SSIM和医学特定指标。

### Q: 模型可以用于其他医学任务吗？
A: 可以，但需要：
1. 调整数据预处理
2. 修改损失函数
3. 重新训练模型

### Q: 如何部署模型到临床环境？
A:
1. 优化模型推理速度
2. 添加输入验证
3. 实现批量处理
4. 添加日志和监控

## 技术支持

如有问题，请：
1. 查看项目GitHub Issues
2. 参考原始CycleGAN论文
3. 检查医学图像质量评估文献
4. 联系项目维护者

## 引用

如果您使用了本项目，请引用：

```bibtex
@article{zhu2017unpaired,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  journal={International Conference on Computer Vision (ICCV)},
  year={2017}
}
```