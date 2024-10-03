from Models.Alex import alex_cifar10
from Models.init_model import CNNMnist, LogisticRegression
from Models.resnet import resnet18, resnet34, resnet50
from Models.resnet_cifar import ResNet18
from Models.simple import SimpleNet, CNNCifar
from Models.vgg import VGG


def init_model(cfg):
    model = cfg.model
    classes = cfg.classes
    if model == 'lr':
        net = LogisticRegression(784, 10)
    elif model == 'cnn':
        net = CNNMnist(10)
    elif model == 'simple':
        net = SimpleNet(10)
    elif model == 'CNNCifar':
        net = CNNCifar()
    elif model == 'alex_cifar':
        net = alex_cifar10()
    elif model == 'resnet18':
        net = ResNet18(num_classes=classes)
    elif model == 'resnet34':
        net = resnet34(num_classes=classes)
    elif model == 'resnet50':
        net = resnet50(num_classes=classes)
    elif model == 'dba_resnet18':
        net = ResNet18(name="resnet18")
    elif model == 'vgg11':
        net = VGG('VGG11', num_classes=classes)
    elif model == 'vgg16':
        net = VGG('VGG16', num_classes=classes)
    else:
        assert False, "Invalid model"
    return net
