import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.model import Model

def compute_gaussian_noise(data, sigma, s):
    shape = data.shape
    noise = Variable(torch.zeros(shape))
    noise.data.normal_(0.0, std=sigma*sigma)
    return noise.to('cuda:0')

class SimpleNet(Model):
    def __init__(self, num_classes):
        super().__init__()
        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        if num_classes == 10:
            self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
            self.fc1 = nn.Linear(8 * 8 * 64, 512)
            self.fc2 = nn.Linear(512, num_classes)
        elif num_classes == 200:
            self.conv1 = nn.Conv2d(3, 24, kernel_size=(3, 3), stride=(1, 1), padding=1)
            self.conv2 = nn.Conv2d(24, 50, kernel_size=(3, 3), stride=(1, 1), padding=1)
            self.fc1 = nn.Linear(16 * 16 * 50, 1024)
            self.fc2 = nn.Linear(1024, num_classes)

        # self.fc3 = nn.Linear(32, num_classes)

    def first_activations(self, x):
        x = F.relu(self.conv1(x))
        return x

    def final_activations(self, x, layer=None):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        return x

    # return the activations after the last conv_layers
    def features(self, x, layer=None):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size()[0], -1)

        return x

    def forward(self, x, latent=False):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        if x.requires_grad:
            x.register_hook(self.activations_hook)

        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        # out = F.log_softmax(x, dim=1)
        # print("outsize:",out.size())
        if latent:
            return out, x
        else:
            return out


class CNNCifar(nn.Module):
    def __init__(self):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3,   64,  3)
        self.conv2 = nn.Conv2d(64,  128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
    def add_noise(self, sigma, s):
        with torch.no_grad():
            for param in self.parameters():
                noise = compute_gaussian_noise(param.data, sigma, s)
                param += noise

