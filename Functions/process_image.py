import os

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms


def tensor2image(tensor):
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  ###去掉batch维度
    if tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0)  ##将c,h,w 转换为h,w,c
    # tensor = tensor.mul(255).clamp(0, 255)  ###将像素值转换为0-255之间
    tensor = tensor.mul(255.0).clamp(0.0, 255.0)

    tensor = tensor.cpu().numpy().astype('uint8')  ###
    toPil = transforms.ToPILImage()
    img = toPil(tensor)
    return img


def image2tensor(pic_path):
    # 加载图像
    image = Image.open(pic_path)  # 替换为您的图像文件路径
    # 定义转换操作
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    # 将图像转换为Tensor
    tensor = transform(image)
    return tensor


def tensor2normalize(img, DEVICE='cuda:0',dataset='cifar10'):
    img = img.to(DEVICE)

    if dataset == 'mnist':
        mean = [0.5]
        std = [0.5]
    elif dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif dataset == 'tiny-imagenet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        assert (2 == 1)
    t_mean = torch.FloatTensor(mean).view(img.shape[0], 1, 1).expand(img.shape).to(DEVICE)
    t_std = torch.FloatTensor(std).view(img.shape[0], 1, 1).expand(img.shape).to(DEVICE)
    img = (img - t_mean) / t_std
    return img


def tensor2denormalize(img, DEVICE='cuda:0',dataset='cifar10'):
    img = img.to(DEVICE)

    if dataset == 'mnist':
        mean = [0.5]
        std = [0.5]
    elif dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif dataset == 'tiny-imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        assert (2 == 1)
    t_mean = torch.FloatTensor(mean).view(img.shape[0], 1, 1).expand(img.shape).to(DEVICE)
    t_std = torch.FloatTensor(std).view(img.shape[0], 1, 1).expand(img.shape).to(DEVICE)
    img = img * t_std + t_mean
    return img


def tensor2Img(tensor):
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  ###去掉batch维度
    if tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0)  ##将c,h,w 转换为h,w,c
    # tensor = tensor.mul(255).clamp(0, 255)  ###将像素值转换为0-255之间
    tensor = tensor.mul(255.0).clamp(0.0, 255.0)

    tensor = tensor.cpu().numpy().astype('uint8')  ###
    toPil = transforms.ToPILImage()
    img = toPil(tensor)
    return img

def Img2Tensor(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    # 将图像转换为Tensor
    tensor = transform(img)
    return tensor

def getImg2Tensor(pic_path):
    # 加载图像
    image = Image.open(pic_path)  # 替换为您的图像文件路径
    tensor = Img2Tensor(image)
    return tensor


def saveTensor2Img(img, savepath):
    img = tensor2Img(img)
    img.save(savepath)



# 存储多图可视化
def save_multi_image(images, save_dir='./save_dir', name='./', n_row=10, n_column=10, hspace=0.1, wspace=0.1, scale=2,
                     param={'x_label': 'x', 'y_label': 'y'}):
    param = {
        'Title': ['1', '2', '3', '4', '5', '6', '7', '8', '9','10','mean'],
        'y_label': ['target']
    }


    # 创建一个新的10x10的子图网格
    fig, axes = plt.subplots(n_row, n_column, figsize=(n_column * scale, n_row * scale))

    # 设置子图之间的水平和垂直间隔
    fig.subplots_adjust(hspace=hspace, wspace=wspace)
    # print(len(axes.flat),n_row,n_column)
    # 遍历每个子图，绘制一张随机彩色图像，并添加行列信息
    for i, img in enumerate(images):

        # 获取当前子图
        ax = axes.flat[i]

        # 绘制图像
        img = np.clip(img, 0, 1)
        img = img.transpose(1, 2, 0)
        ax.imshow(img)
        ax.axis('auto')
        ax.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)

        # 添加行列信息
        row, col = divmod(i, n_column)  # 计算当前图像的行列位置
        # if row == n_row - 1:
        #     ax.set_xlabel(param['x_label'], fontsize=12, position=(0.5, -0.1))
        # if col == 0:
        #     ax.set_title(param['Title'], fontsize=12, position=(-0.1, 0.5))

    plt.show()
    plt.savefig(os.path.join(save_dir, name))


