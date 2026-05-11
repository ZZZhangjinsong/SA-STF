import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader
import numpy as np


# FeatureHook 类用于提取VGG层的特征
class FeatureHook:
    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self.on)

    def on(self, module, inputs, outputs):
        self.features = outputs

    def close(self):
        self.hook.remove()


# 感知损失函数
def perceptual_loss(x, y):
    return F.mse_loss(x, y)


# PerceptualLoss 类用于计算感知损失
class PerceptualLoss(nn.Module):
    def __init__(self, blocks, weights, device):
        super(PerceptualLoss, self).__init__()
        assert len(blocks) == len(weights)

        self.weights = torch.tensor(weights).to(device)

        # 加载VGG16模型（使用BN版本）
        vgg = models.vgg16_bn(pretrained=True).features
        vgg.eval()

        # 禁用梯度计算
        for param in vgg.parameters():
            param.requires_grad = False

        vgg = vgg.to(device)

        # 获取VGG16中不同块的位置，VGG16共有5个块
        bns = [i - 2 for i, m in enumerate(vgg) if isinstance(m, nn.MaxPool2d)]
        assert all(isinstance(vgg[bn], nn.BatchNorm2d) for bn in bns)

        # 创建特征提取钩子
        self.hooks = [FeatureHook(vgg[bns[i]]) for i in blocks]
        self.features = vgg[0: bns[blocks[-1]] + 1]

    def forward(self, inputs, targets):
        inputs = inputs.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)
        targets = targets.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)
        # 手动标准化，符合VGG16的要求（ImageNet的均值和标准差）
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        # 按照每个通道进行归一化
        inputs = (inputs - torch.tensor(mean).view(1, 3, 1, 1).to(inputs.device)) / torch.tensor(std).view(1, 3, 1,
                                                                                                           1).to(
            inputs.device)
        targets = (targets - torch.tensor(mean).view(1, 3, 1, 1).to(targets.device)) / torch.tensor(std).view(1, 3, 1,
                                                                                                              1).to(
            targets.device)

        # 提取图像特征
        self.features(inputs)
        input_features = [hook.features.clone() for hook in self.hooks]

        self.features(targets)
        target_features = [hook.features for hook in self.hooks]

        loss = 0.0

        # 计算加权的感知损失
        for lhs, rhs, w in zip(input_features, target_features, self.weights):
            lhs = lhs.view(lhs.size(0), -1)  # 展平
            rhs = rhs.view(rhs.size(0), -1)  # 展平
            loss += perceptual_loss(lhs, rhs) * w

        return loss





# 测试代码：生成随机的6通道图像并计算感知损失
if __name__ == '__main__':
    # 设置设备（如果有CUDA，则使用GPU）
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

    # 初始化 PerceptualLoss，选择前三个块，并设置相应的权重
    blocks = [0, 1, 2]  # 使用VGG16的前3个卷积块
    weights = [1.0, 0.8, 0.6]  # 为每个块设置不同的权重
    perceptual_loss_fn = PerceptualLoss(blocks, weights, device)

    # 随机生成两张6通道图像，大小为 6x256x256
    image1 = torch.rand(1, 6, 256, 256).to(device)
    image2 = torch.rand(1, 6, 256, 256).to(device)

    # 计算感知损失
    loss = perceptual_loss_fn(image1, image2)

    print(f"Perceptual Loss: {loss.item()}")

