from torch import nn

from Models.Alex import alex_cifar10
from Models.init_model import CNNMnist, LogisticRegression
from Models.resnet import resnet34, resnet50
from Models.simple import SimpleNet, CNNCifar
# from Models.vgg import VGG
from Models.vgg_model import VGG
from Models.resnet_cifar import ResNet18
# from models_test.resnet import resnet18 as ResNet18


def init_model(cfg):
    model = cfg.model
    classes = cfg.classes
    if model == 'lr':
        net = LogisticRegression(784, 10)
    # elif model == 'cnn':
    #     net = CNNMnist(10)
    elif model == 'cnn':
        net = SimpleNet(10)
    elif model == 'CNNCifar':
        net = CNNCifar()
    elif model == 'alex_cifar':
        net = alex_cifar10()
    elif model == 'resnet18':
        net = ResNet18(num_classes=classes)
        net.avgpool_name = cfg.dataset
        if cfg.dataset == 'tiny-imagenet':
            net.avgpool = nn.AdaptiveAvgPool2d(1)
            net.linear = nn.Linear(256, 200)

    elif model == 'resnet34':
        net = resnet34(num_classes=classes)
    elif model == 'resnet50':
        net = resnet50(num_classes=classes)
    elif model == 'dba_resnet18':
        net = ResNet18(name="resnet18")
    elif model == 'vgg11':
        net = VGG('VGG11', num_classes=classes, channels=3)
    elif model == 'vgg16':
        net = VGG('VGG16', num_classes=classes)
    else:
        assert False, "Invalid model"
    return net
