"""
@author: Manaar Alam
"""
from collections import OrderedDict  # 导入 OrderedDict

import torch.nn as nn
import torch.nn.functional as F
import utils

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes, channels):
        super(VGG, self).__init__()
        self.in_channels = channels
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        output = F.log_softmax(out, dim=1)
        return output

    # 新增方法，返回第一卷积层后的激活值
    def first_activations(self, x):
        # 提取特征中的前几层
        first_conv = self.features[0]  # 第1个卷积层
        first_bn = self.features[1]  # 第1个批归一化层
        first_relu = self.features[2]  # 第1个ReLU激活层

        # 计算经过第一卷积层、归一化层、ReLU激活后的输出
        x = first_conv(x)
        x = first_bn(x)
        x = first_relu(x)
        return x

    # def _make_layers(self, cfg):
    #     layers = []
    #     in_channels = self.in_channels
    #     for x in cfg:
    #         if x == 'M':
    #             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    #         else:
    #             layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
    #                     nn.BatchNorm2d(x),
    #                     nn.ReLU(inplace=True)]
    #             in_channels = x
    #     layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
    #     return nn.Sequential(*layers)

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.in_channels
        for idx, x in enumerate(cfg):
            if x == 'M':
                layers.append(('pool' + str(idx), nn.MaxPool2d(kernel_size=2, stride=2)))
            else:
                layers.append(('conv' + str(idx), nn.Conv2d(in_channels, x, kernel_size=3, padding=1)))
                layers.append(('bn' + str(idx), nn.BatchNorm2d(x)))
                layers.append(('relu' + str(idx), nn.ReLU(inplace=True)))
                in_channels = x
        layers.append(('avgpool', nn.AvgPool2d(kernel_size=1, stride=1)))
        return nn.Sequential(OrderedDict(layers))  # 使用OrderedDict以便按名称访问

