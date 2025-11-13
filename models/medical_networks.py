import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from . import networks

class MedicalResnetGenerator(nn.Module):
    """医学图像增强专用ResNet生成器"""

    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9, norm_layer=nn.InstanceNorm2d,
                 use_dropout=False, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(MedicalResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        # 初始卷积层 - 更大的感受野
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        # 下采样层
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.ReflectionPad2d(1),
                     nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0, bias=True),
                     norm_layer(ngf * mult * 2),
                     nn.ReLU(True)]

        # ResNet块
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResNetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                 use_dropout=use_dropout, use_bias=True)]

        # 上采样层
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                     nn.ReflectionPad2d(1),
                     nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, padding=0, bias=True),
                     norm_layer(int(ngf * mult / 2)),
                     nn.ReLU(True)]

        # 输出层
        model += [nn.ReflectionPad2d(3),
                 nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0, bias=True),
                 nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class ResNetBlock(nn.Module):
    """定义ResNet块"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResNetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                      norm_layer(dim),
                      nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                      norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class MedicalDiscriminator(nn.Module):
    """医学图像增强专用判别器 - 注意边缘和纹理细节"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        super(MedicalDiscriminator, self).__init__()

        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """标准前向传播"""
        return self.model(input)


class MedicalEnhancementLoss(nn.Module):
    """医学图像增强专用损失函数"""

    def __init__(self):
        super(MedicalEnhancementLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

        # Sobel算子用于边缘检测
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def get_edges(self, img):
        """获取图像边缘"""
        if img.dim() == 3:
            img = img.unsqueeze(0)
        if img.size(1) == 3:
            # 转换为灰度
            img = 0.299 * img[:, 0:1, :, :] + 0.587 * img[:, 1:2, :, :] + 0.114 * img[:, 2:3, :, :]

        edge_x = F.conv2d(img, self.sobel_x, padding=1)
        edge_y = F.conv2d(img, self.sobel_y, padding=1)
        return torch.sqrt(edge_x**2 + edge_y**2)

    def edge_preservation_loss(self, pred, target):
        """边缘保持损失"""
        pred_edges = self.get_edges(pred)
        target_edges = self.get_edges(target)
        return self.l1_loss(pred_edges, target_edges)

    def structural_similarity(self, pred, target, window_size=11, sigma=1.5):
        """计算结构相似性"""
        # 简化的SSIM实现
        mu_pred = F.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
        mu_target = F.avg_pool2d(target, window_size, stride=1, padding=window_size//2)

        sigma_pred = F.avg_pool2d(pred**2, window_size, stride=1, padding=window_size//2) - mu_pred**2
        sigma_target = F.avg_pool2d(target**2, window_size, stride=1, padding=window_size//2) - mu_target**2
        sigma_pred_target = F.avg_pool2d(pred*target, window_size, stride=1, padding=window_size//2) - mu_pred*mu_target

        c1 = 0.01**2
        c2 = 0.03**2

        ssim = ((2*mu_pred*mu_target + c1)*(2*sigma_pred_target + c2)) / \
               ((mu_pred**2 + mu_target**2 + c1)*(sigma_pred + sigma_target + c2))

        return ssim.mean()

    def contrast_enhancement_loss(self, pred, target):
        """对比度增强损失"""
        # 计算局部对比度
        def local_contrast(img):
            mean = F.avg_pool2d(img, 5, stride=1, padding=2)
            var = F.avg_pool2d((img - mean)**2, 5, stride=1, padding=2)
            return var

        pred_contrast = local_contrast(pred)
        target_contrast = local_contrast(target)

        return self.l1_loss(pred_contrast, target_contrast)

    def forward(self, pred, target, alpha=1.0, beta=0.1, gamma=0.05):
        """计算总损失"""
        l1_loss = self.l1_loss(pred, target)

        # SSIM损失（转换为最小化问题）
        ssim_loss = 1.0 - self.structural_similarity(pred, target)

        # 边缘保持损失
        edge_loss = self.edge_preservation_loss(pred, target)

        # 对比度增强损失
        contrast_loss = self.contrast_enhancement_loss(pred, target)

        total_loss = alpha * l1_loss + beta * ssim_loss + gamma * edge_loss + 0.02 * contrast_loss

        return total_loss, {
            'L1': l1_loss.item(),
            'SSIM': ssim_loss.item(),
            'Edge': edge_loss.item(),
            'Contrast': contrast_loss.item()
        }