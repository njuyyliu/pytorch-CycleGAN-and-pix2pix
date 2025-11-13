import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .medical_networks import MedicalResnetGenerator, MedicalDiscriminator, MedicalEnhancementLoss


class MedicalCycleGANModel(BaseModel):
    """
    医学图像增强专用CycleGAN模型
    针对医学图像的特殊需求进行了优化
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """添加医学图像特定的命令行选项"""
        parser.set_defaults(no_dropout=True)  # 医学图像通常不使用dropout

        if is_train:
            parser.add_argument("--lambda_A", type=float, default=10.0, help="前向循环损失权重")
            parser.add_argument("--lambda_B", type=float, default=10.0, help="后向循环损失权重")
            parser.add_argument("--lambda_identity", type=float, default=0.5, help="身份映射损失权重")
            parser.add_argument("--lambda_medical", type=float, default=1.0, help="医学图像专用损失权重")
            parser.add_argument("--use_medical_loss", action='store_true', help="是否使用医学图像专用损失")
            parser.add_argument("--preserve_structure", action='store_true', help="是否保留结构信息")
            parser.add_argument("--enhance_contrast", action='store_true', help="是否增强对比度")

        return parser

    def __init__(self, opt):
        """初始化医学图像增强CycleGAN模型"""
        BaseModel.__init__(self, opt)

        # 损失函数名称
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']

        if getattr(opt, 'use_medical_loss', False):
            self.loss_names.extend(['medical_A', 'medical_B'])

        # 可视化图像名称
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']

        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B

        # 模型名称
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:
            self.model_names = ['G_A', 'G_B']

        # 定义网络 - 使用医学图像专用架构
        self.netG_A = MedicalResnetGenerator(
            opt.input_nc, opt.output_nc, opt.ngf,
            n_blocks=9, norm_layer=networks.get_norm_layer(opt.norm),
            use_dropout=not opt.no_dropout
        )

        self.netG_B = MedicalResnetGenerator(
            opt.output_nc, opt.input_nc, opt.ngf,
            n_blocks=9, norm_layer=networks.get_norm_layer(opt.norm),
            use_dropout=not opt.no_dropout
        )

        if self.isTrain:
            self.netD_A = MedicalDiscriminator(
                opt.output_nc, opt.ndf,
                n_layers=getattr(opt, 'n_layers_D', 3),
                norm_layer=networks.get_norm_layer(opt.norm)
            )

            self.netD_B = MedicalDiscriminator(
                opt.input_nc, opt.ndf,
                n_layers=getattr(opt, 'n_layers_D', 3),
                norm_layer=networks.get_norm_layer(opt.norm)
            )

        if self.isTrain:
            if opt.lambda_identity > 0.0:
                assert(opt.input_nc == opt.output_nc)

            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)

            # 定义损失函数
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()

            # 医学图像专用损失函数
            if getattr(opt, 'use_medical_loss', False):
                self.criterionMedical = MedicalEnhancementLoss().to(self.device)

            # 初始化优化器
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999)
            )

            self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999)
            )

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """设置输入数据"""
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def forward(self):
        """前向传播"""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A) -> B
        self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A)) -> A
        self.fake_A = self.netG_B(self.real_B)  # G_B(B) -> A
        self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B)) -> B

    def backward_D_basic(self, netD, real, fake):
        """判别器基本反向传播"""
        # 真实损失
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)

        # 伪造损失
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)

        # 总损失
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()

        return loss_D

    def backward_D_A(self):
        """判别器A反向传播"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """判别器B反向传播"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """生成器反向传播"""
        # 身份损失
        if self.opt.lambda_identity > 0.0:
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * self.opt.lambda_B * self.opt.lambda_identity
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * self.opt.lambda_A * self.opt.lambda_identity
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN损失
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        # 循环损失
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * self.opt.lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * self.opt.lambda_B

        # 医学图像专用损失
        if getattr(self.opt, 'use_medical_loss', False):
            self.loss_medical_A, _ = self.criterionMedical(self.fake_B, self.real_B, alpha=self.opt.lambda_medical)
            self.loss_medical_B, _ = self.criterionMedical(self.fake_A, self.real_A, alpha=self.opt.lambda_medical)
        else:
            self.loss_medical_A = 0
            self.loss_medical_B = 0

        # 总损失
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + \
                      self.loss_idt_A + self.loss_idt_B + self.loss_medical_A + self.loss_medical_B

        self.loss_G.backward()

    def optimize_parameters(self):
        """优化参数"""
        # 前向传播
        self.forward()

        # 更新G
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # 更新D
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()