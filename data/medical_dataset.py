import os
import torch
import numpy as np
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import pydicom
import cv2

class MedicalEnhancementDataset(BaseDataset):
    """
    医学图像增强专用数据集类
    支持DICOM和标准图像格式
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """添加医学图像特定的命令行选项"""
        parser.add_argument('--medical_modality', type=str, default='CT',
                           choices=['CT', 'MRI', 'XRay'], help='医学成像模态')
        parser.add_argument('--window_width', type=float, default=400, help='CT窗宽')
        parser.add_argument('--window_level', type=float, default=40, help='CT窗位')
        parser.add_argument('--use_dicom', action='store_true', help='是否使用DICOM格式')
        return parser

    def __init__(self, opt):
        """初始化医学图像数据集"""
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + "A")  # 低质量图像
        self.dir_B = os.path.join(opt.dataroot, opt.phase + "B")  # 高质量图像

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        # 医学图像特定参数
        self.modality = opt.medical_modality
        self.window_width = opt.window_width
        self.window_level = opt.window_level
        self.use_dicom = opt.use_dicom

        btoA = self.opt.direction == "BtoA"
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc

        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def load_dicom_image(self, path):
        """加载DICOM图像并应用窗宽窗位"""
        try:
            ds = pydicom.dcmread(path)
            img = ds.pixel_array.astype(np.float32)

            if self.modality == 'CT':
                # 应用CT窗宽窗位
                img = self.apply_window_level(img, self.window_width, self.window_level)

            # 归一化到0-255
            img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

            # 转换为PIL图像
            if len(img.shape) == 2:
                return Image.fromarray(img, mode='L')
            else:
                return Image.fromarray(img)

        except Exception as e:
            print(f"Error loading DICOM {path}: {e}")
            # 回退到普通图像加载
            return Image.open(path).convert('L')

    def apply_window_level(self, img, window_width, window_level):
        """应用CT窗宽窗位"""
        min_val = window_level - window_width // 2
        max_val = window_level + window_width // 2
        windowed_img = np.clip(img, min_val, max_val)
        return windowed_img

    def medical_augmentation(self, img):
        """医学图像特定的数据增强"""
        # 转换为numpy数组
        if isinstance(img, Image.Image):
            img = np.array(img)

        # 1. 对比度增强
        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2)
            img = np.clip(img * alpha, 0, 255).astype(np.uint8)

        # 2. 轻微高斯噪声（模拟医学图像噪声）
        if random.random() > 0.7:
            noise = np.random.normal(0, 2, img.shape)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)

        # 3. 小幅度旋转
        if random.random() > 0.8:
            angle = random.uniform(-5, 5)
            h, w = img.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h))

        return Image.fromarray(img)

    def __getitem__(self, index):
        """获取数据样本"""
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[random.randint(0, self.B_size - 1)] if self.A_size != self.B_size else self.B_paths[index % self.B_size]

        # 加载图像
        if self.use_dicom and (A_path.endswith('.dcm') or B_path.endswith('.dcm')):
            A_img = self.load_dicom_image(A_path)
            B_img = self.load_dicom_image(B_path)
        else:
            A_img = Image.open(A_path).convert('RGB')
            B_img = Image.open(B_path).convert('RGB')

        # 数据增强（仅训练时）
        if self.isTrain and not self.opt.no_augment:
            A_img = self.medical_augmentation(A_img)
            B_img = self.medical_augmentation(B_img)

        # 应用变换
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """返回数据集大小"""
        return max(self.A_size, self.B_size)