"""ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from Models.init_model import CNNMnist


class SimpleNet(nn.Module):
    def __init__(self, name=None, created_time=None):
        super(SimpleNet, self).__init__()
        self.created_time = created_time
        self.name = name

    def save_stats(self, epoch, loss, acc):
        self.stats["epoch"].append(epoch)
        self.stats["loss"].append(loss)
        self.stats["acc"].append(acc)

    def copy_params(self, state_dict, coefficient_transfer=100):
        own_state = self.state_dict()

        for name, param in state_dict.items():
            if name in own_state:
                shape = param.shape
                own_state[name].copy_(param.clone())

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(SimpleNet):
    def __init__(self, block, num_blocks, num_classes=10, name=None, created_time=None):
        super(ResNet, self).__init__(name, created_time)
        self.in_planes = 32

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        self.avgpool_name = None

        self.linear = nn.Linear(256 * block.expansion, num_classes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # out = self.avgpool(out)
        if self.avgpool_name == 'tiny-imagenet':
            out = self.avgpool(out)
        elif self.avgpool_name == 'cifar10' or self.avgpool_name == 'mnist':
            out = F.avg_pool2d(out, 4)
        else:
            raise ValueError("avgpool is not supported")
        # out = out.view(out, -1)
        out = out.view(out.size(0), -1)

        out = self.linear(out)
        # for SDTdata
        # return F.softmax(out, dim=1)
        # for regular output
        return out
    def features(self, x):
        # out1 = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        out1 = self.relu(self.bn1(self.conv1(x)))

        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        out5 = out5.view(out5.size()[0], -1)

        return out5
    def first_activations(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        return x

def ResNet18(name=None, created_time=None,num_classes=None):
    return ResNet(
        BasicBlock,
        [2, 2, 2, 2],
        name="{0}_ResNet_18".format(name),
        created_time=created_time,
        num_classes=num_classes
    )


def ResNet34(name=None, created_time=None):
    return ResNet(
        BasicBlock,
        [3, 4, 6, 3],
        name="{0}_ResNet_34".format(name),
        created_time=created_time,
    )


def ResNet50(name=None, created_time=None):
    return ResNet(
        Bottleneck,
        [3, 4, 6, 3],
        name="{0}_ResNet_50".format(name),
        created_time=created_time,
    )


def ResNet101(name=None, created_time=None):
    return ResNet(
        Bottleneck,
        [3, 4, 23, 3],
        name="{0}_ResNet".format(name),
        created_time=created_time,
    )


def ResNet152(name=None, created_time=None):
    return ResNet(
        Bottleneck,
        [3, 8, 36, 3],
        name="{0}_ResNet".format(name),
        created_time=created_time,
    )


def layer2module(model, layer: str):
    if isinstance(model, ResNet):
        if layer == 'conv1.weight':
            return 'relu'
        elif 'conv1.weight' in layer:
            return layer.replace('conv1.weight', 'relu')
        elif 'conv2.weight' in layer:
            return layer.replace('.conv2.weight', '')
    elif isinstance(model, SimpleNet) or isinstance(model, CNNMnist):
        module_name = None
        if 'conv' in layer:
            module_name = layer.split('.')[0]
        elif 'fc' in layer:
            module_name = layer.split('.')[0]
        return module_name

if __name__ == "__main__":
    net = ResNet18()
    y = net(Variable(torch.randn(1, 3, 32, 32)))
    print(y.size())
