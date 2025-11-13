import torch
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd


class MedicalImageEvaluator:
    """医学图像质量评估器"""

    def __init__(self):
        self.metrics_history = []

    def calculate_psnr(self, pred, target, max_val=1.0):
        """计算PSNR"""
        return psnr(target, pred, data_range=max_val)

    def calculate_ssim(self, pred, target):
        """计算结构相似性"""
        return ssim(target, pred, multichannel=True if pred.shape[-1] > 1 else False)

    def calculate_entropy(self, image):
        """计算图像熵（信息量）"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        hist = hist[hist > 0]  # 移除零值
        entropy = -np.sum(hist * np.log2(hist))
        return entropy

    def calculate_contrast_measure(self, image):
        """计算对比度度量（RMS对比度）"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        mean_intensity = np.mean(gray)
        contrast = np.sqrt(np.mean((gray - mean_intensity) ** 2))
        return contrast

    def calculate_edge_density(self, image):
        """计算边缘密度"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        return edge_density

    def calculate_noise_level(self, image):
        """估算噪声水平"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # 使用拉普拉斯算子估算噪声
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_level = np.var(laplacian)
        return noise_level

    def calculate_mtf_measure(self, edge_image):
        """计算调制传递函数（MTF）近似值"""
        if len(edge_image.shape) == 3:
            gray = cv2.cvtColor(edge_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = edge_image

        # 找到边缘
        edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)

        # 计算边缘强度作为MTF的近似
        mtf_measure = np.mean(gray[edges > 0]) / 255.0
        return mtf_measure

    def evaluate_single_pair(self, pred, target, name=""):
        """评估单对图像"""
        metrics = {
            'name': name,
            'PSNR': self.calculate_psnr(pred, target),
            'SSIM': self.calculate_ssim(pred, target),
            'Pred_Entropy': self.calculate_entropy(pred),
            'Target_Entropy': self.calculate_entropy(target),
            'Pred_Contrast': self.calculate_contrast_measure(pred),
            'Target_Contrast': self.calculate_contrast_measure(target),
            'Pred_Edge_Density': self.calculate_edge_density(pred),
            'Target_Edge_Density': self.calculate_edge_density(target),
            'Pred_Noise_Level': self.calculate_noise_level(pred),
            'Target_Noise_Level': self.calculate_noise_level(target),
            'Contrast_Improvement': 0,
            'Noise_Reduction': 0,
            'Edge_Preservation': 0
        }

        # 计算改善指标
        metrics['Contrast_Improvement'] = metrics['Pred_Contrast'] / (metrics['Target_Contrast'] + 1e-8)
        metrics['Noise_Reduction'] = metrics['Target_Noise_Level'] / (metrics['Pred_Noise_Level'] + 1e-8)
        metrics['Edge_Preservation'] = 1 - abs(metrics['Pred_Edge_Density'] - metrics['Target_Edge_Density'])

        return metrics

    def evaluate_dataset(self, pred_dir, target_dir, output_dir="./evaluation_results"):
        """评估整个数据集"""
        pred_paths = sorted(Path(pred_dir).glob("*.png")) + sorted(Path(pred_dir).glob("*.jpg"))
        target_paths = sorted(Path(target_dir).glob("*.png")) + sorted(Path(target_dir).glob("*.jpg"))

        print(f"Found {len(pred_paths)} prediction images and {len(target_paths)} target images")

        all_metrics = []

        for pred_path, target_path in zip(pred_paths, target_paths):
            if pred_path.stem == target_path.stem:
                pred_img = plt.imread(pred_path)
                target_img = plt.imread(target_path)

                # 归一化到[0,1]
                pred_img = pred_img.astype(np.float32) / 255.0
                target_img = target_img.astype(np.float32) / 255.0

                metrics = self.evaluate_single_pair(pred_img, target_img, pred_path.stem)
                all_metrics.append(metrics)

        # 保存详细结果
        df = pd.DataFrame(all_metrics)
        df.to_csv(f"{output_dir}/detailed_metrics.csv", index=False)

        # 计算平均指标
        avg_metrics = df.mean(numeric_only=True)
        std_metrics = df.std(numeric_only=True)

        # 创建报告
        self.create_evaluation_report(avg_metrics, std_metrics, output_dir)

        return avg_metrics, std_metrics

    def create_evaluation_report(self, avg_metrics, std_metrics, output_dir):
        """创建评估报告"""
        Path(output_dir).mkdir(exist_ok=True)

        # 保存平均指标
        report = f"""
医学图像增强评估报告
================

质量指标:
- PSNR: {avg_metrics['PSNR']:.3f} ± {std_metrics['PSNR']:.3f} dB
- SSIM: {avg_metrics['SSIM']:.3f} ± {std_metrics['SSIM']:.3f}

图像特征指标:
- 预测图像熵: {avg_metrics['Pred_Entropy']:.3f} ± {std_metrics['Pred_Entropy']:.3f}
- 目标图像熵: {avg_metrics['Target_Entropy']:.3f} ± {std_metrics['Target_Entropy']:.3f}

对比度指标:
- 预测对比度: {avg_metrics['Pred_Contrast']:.3f} ± {std_metrics['Pred_Contrast']:.3f}
- 目标对比度: {avg_metrics['Target_Contrast']:.3f} ± {std_metrics['Target_Contrast']:.3f}
- 对比度改善比例: {avg_metrics['Contrast_Improvement']:.3f} ± {std_metrics['Contrast_Improvement']:.3f}

边缘指标:
- 预测边缘密度: {avg_metrics['Pred_Edge_Density']:.3f} ± {std_metrics['Pred_Edge_Density']:.3f}
- 目标边缘密度: {avg_metrics['Target_Edge_Density']:.3f} ± {std_metrics['Target_Edge_Density']:.3f}
- 边缘保持度: {avg_metrics['Edge_Preservation']:.3f} ± {std_metrics['Edge_Preservation']:.3f}

噪声指标:
- 预测噪声水平: {avg_metrics['Pred_Noise_Level']:.3f} ± {std_metrics['Pred_Noise_Level']:.3f}
- 目标噪声水平: {avg_metrics['Target_Noise_Level']:.3f} ± {std_metrics['Target_Noise_Level']:.3f}
- 噪声抑制比例: {avg_metrics['Noise_Reduction']:.3f} ± {std_metrics['Noise_Reduction']:.3f}
"""

        with open(f"{output_dir}/evaluation_report.txt", "w", encoding="utf-8") as f:
            f.write(report)

        # 创建可视化图表
        self.create_visualization(avg_metrics, std_metrics, output_dir)

        print(report)

    def create_visualization(self, avg_metrics, std_metrics, output_dir):
        """创建可视化图表"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # PSNR和SSIM
        quality_metrics = ['PSNR', 'SSIM']
        quality_values = [avg_metrics[metric] for metric in quality_metrics]
        quality_stds = [std_metrics[metric] for metric in quality_metrics]

        axes[0, 0].bar(quality_metrics, quality_values, yerr=quality_stds, capsize=5)
        axes[0, 0].set_title('图像质量指标')
        axes[0, 0].set_ylabel('值')

        # 对比度改善
        improvement_metrics = ['Contrast_Improvement', 'Noise_Reduction', 'Edge_Preservation']
        improvement_values = [avg_metrics[metric] for metric in improvement_metrics]
        improvement_stds = [std_metrics[metric] for metric in improvement_metrics]

        axes[0, 1].bar(improvement_metrics, improvement_values, yerr=improvement_stds, capsize=5)
        axes[0, 1].set_title('图像改善指标')
        axes[0, 1].set_ylabel('改善比例')
        axes[0, 1].axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='基准线')
        axes[0, 1].legend()

        # 图像特征对比
        feature_metrics = ['Pred_Entropy', 'Target_Entropy', 'Pred_Contrast', 'Target_Contrast']
        feature_labels = ['预测熵', '目标熵', '预测对比度', '目标对比度']
        feature_values = [avg_metrics[metric] for metric in feature_metrics]

        bars = axes[1, 0].bar(feature_labels, feature_values)
        axes[1, 0].set_title('图像特征对比')
        axes[1, 0].set_ylabel('值')

        # 为预测和目标图像添加不同颜色
        bars[0].set_color('blue')
        bars[1].set_color('red')
        bars[2].set_color('blue')
        bars[3].set_color('red')

        # 边缘和噪声指标
        edge_noise_metrics = ['Pred_Edge_Density', 'Target_Edge_Density', 'Pred_Noise_Level', 'Target_Noise_Level']
        edge_noise_labels = ['预测边缘密度', '目标边缘密度', '预测噪声水平', '目标噪声水平']
        edge_noise_values = [avg_metrics[metric] for metric in edge_noise_metrics]

        bars = axes[1, 1].bar(edge_noise_labels, edge_noise_values)
        axes[1, 1].set_title('边缘和噪声指标')
        axes[1, 1].set_ylabel('值')

        bars[0].set_color('blue')
        bars[1].set_color('red')
        bars[2].set_color('blue')
        bars[3].set_color('red')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/evaluation_visualization.png", dpi=300, bbox_inches='tight')
        plt.close()


def create_enhancement_comparison(original, enhanced, save_path):
    """创建增强效果对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 原始图像
    axes[0].imshow(original, cmap='gray' if len(original.shape) == 2 else None)
    axes[0].set_title('原始图像')
    axes[0].axis('off')

    # 增强图像
    axes[1].imshow(enhanced, cmap='gray' if len(enhanced.shape) == 2 else None)
    axes[1].set_title('增强图像')
    axes[1].axis('off')

    # 差异图
    diff = np.abs(enhanced - original)
    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title('差异图')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()