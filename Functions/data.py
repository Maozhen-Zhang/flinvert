import copy
import math
import random
import torch
import torch.nn

from collections import defaultdict

import torchvision
from torchvision import datasets, transforms

from torch.utils.data import DataLoader, TensorDataset, Dataset
# 其他模块
import numpy as np

from Functions.TinyImageNet import TinyImagenetFederatedTask


def get_dataset(dataset='', root_path='', IsNormalize=False):
    # dataset = conf['dataset']
    # root_path = conf['root_path']
    # IsNormalize = conf['Normalize']
    if dataset == 'mnist':
        path = root_path + '/'
        # 因为resnet18输入的CHW是(3, 224, 224)，而mnist中单张图片CHW是(1, 28, 28)，所以需要对MNIST数据集进行预处理。

        # transform = transforms.Compose([transforms.Resize((224, 224)), transforms.Grayscale(3), transforms.ToTensor(),
        #                                 transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)), ])

        # # 定义数据预处理操作
        transform = transforms.Compose([
            # 将图像调整为固定的尺寸，如28x28
            transforms.Resize((28, 28)),
            # 将图像转换为Tensor格式，并进行归一化，将像素值缩放到 [0, 1] 的范围
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        # transform = transforms.Compose([transforms.Resize((224, 224)), transforms.Grayscale(3), transforms.ToTensor(),
        #                                 transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)), ])

        if IsNormalize == False:
            transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.MNIST(path, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(path, train=False, download=True, transform=transform)
    elif dataset == 'femnist':
        path = root_path + '/'

        transform = transforms.Compose([
            # 将图像调整为固定的尺寸，如28x28
            transforms.Resize((28, 28)),
            # 将图像转换为Tensor格式，并进行归一化，将像素值缩放到 [0, 1] 的范围
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        if IsNormalize == False:
            transform = transforms.Compose([transforms.ToTensor()])
            transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.FashionMNIST(path, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(path, train=False, download=True,
                                             transform=transform)

    elif dataset == 'cifar10' or dataset == 'CIFAR10':
        path = root_path + '/CIFAR10'

        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        # transforms.RandomCrop： 切割中心点的位置随机选取
        # transforms.Normalize： 给定均值：(R,G,B) 方差：（R，G，B），将会把Tensor正则化
        transform_train = transforms.Compose(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
             transforms.Normalize(mean, std)])

        transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)])

        if IsNormalize == False:
            transform_train = transforms.Compose([transforms.ToTensor()])
            transform_test = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.CIFAR10(path, train=True, download=True,
                                         transform=transform_train)
        test_dataset = datasets.CIFAR10(path, train=False, download=True,
                                        transform=transform_test)
    elif dataset == 'cifar100' or dataset == 'CIFAR100':
        path = root_path + '/CIFAR100'

        mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        # transforms.RandomCrop： 切割中心点的位置随机选取
        # transforms.Normalize： 给定均值：(R,G,B) 方差：（R，G，B），将会把Tensor正则化
        transform_train = transforms.Compose(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
             transforms.Normalize(mean, std)])

        transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)])

        if IsNormalize == False:
            transform_train = transforms.Compose([transforms.ToTensor()])
            transform_test = transforms.Compose([transforms.ToTensor()])

        train_dataset = datasets.CIFAR100(path, train=True, download=False,
                                          transform=transform_train)
        test_dataset = datasets.CIFAR100(path, train=False, download=False,
                                         transform=transform_test)
    elif dataset == 'tiny-imagenet':
        path = root_path + '/tiny-imagenet-200'
        imagenettask = TinyImagenetFederatedTask()
        train_dataset, test_dataset = imagenettask.load_data()
    else:
        print("Error!!The name is Error!")
        assert (1 == 2)
    return train_dataset, test_dataset


def get_data_indicates(cfg, train_dataset, test_dataset):

    # train_dataset, test_dataset = get_dataset(conf)
    n_users = cfg.n_client
    dataset_slices = []
    non_iid = cfg.heterogenuity
    if non_iid:
        train_indices_all_client = sample_dirichlet_train_data(cfg, train_dataset, n_users, cfg.dirichlet_alpha)
        for id in range(n_users):
            client_dataset_slice = []
            train_indices_per_client = train_indices_all_client[id]
            for i in train_indices_per_client:
                client_dataset_slice.append(train_dataset[i])
            dataset_slices.append(client_dataset_slice)
    else:
        all_range = list(range(len(train_dataset)))
        data_len = int(len(train_dataset) / n_users)
        # SubsetRandomSampler(indices)：会根据indices列表从数据集中按照下标取元素
        # 无放回地按照给定的索引列表采样样本元素。
        random_index = [i for i in range(len(train_dataset))]
        random.shuffle(random_index)

        for id in range(n_users):
            client_dataset_slice = []
            train_indices_per_client = all_range[id * data_len:(id + 1) * data_len]
            for i in train_indices_per_client:
                # client_dataset_slice.append(train_dataset[i])
                client_dataset_slice.append(train_dataset[random_index[i]])

            dataset_slices.append(client_dataset_slice)
    return dataset_slices


def sample_dirichlet_train_data(conf, train_dataset, client_number, alpha):
    cifar_classes = {}
    for indx, x in enumerate(train_dataset):
        _, target = x
        if target in cifar_classes:
            cifar_classes[target].append(indx)
        else:
            cifar_classes[target] = [indx]

    class_size = len(cifar_classes[0])
    list_per_client = defaultdict(list)
    no_classes = len(cifar_classes.keys())

    for n in range(no_classes):
        random.shuffle(cifar_classes[n])
        sampled_probabilities = class_size * np.random.dirichlet(np.array(client_number * [alpha]))
        for user in range(client_number):
            number_of_imgs = int(round(sampled_probabilities[user]))
            sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), number_of_imgs)]
            list_per_client[user].extend(sampled_list)

            cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), number_of_imgs):]

    return list_per_client


def print_distribution(conf, dataset_slices):
    #####################################################################################
    # 打印分布情况
    len_datasets = 0
    for idx, dataset_slice in enumerate(dataset_slices):
        label_counts = [0 for i in range(conf.classes)]
        label_sum = 0
        for group in dataset_slice:
            label_counts[group[1]] += 1
            label_sum += 1
        label_counts[-1] = label_sum
        len_datasets += label_sum
        print(f"|---Client {idx} datasets distribute is {label_counts}")
    print(f"|---Sum datasets lenth is {len_datasets}")


class Subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset) -> None:
        self.dataset = dataset
        # self.indices = indices
        # self.imgs = dataset.data
        # self.labels = np.array(dataset.targets)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[idx]]
        # print(f"|---idx is {idx}")
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def get_val(self, idx):
        # if isinstance(idx, list):
        #     return torch.FloatTensor(self.dataset[i for i in idx][0]), torch.LongTensor(
        #         self.labels[[self.indices[i] for i in idx]])
        return torch.FloatTensor(self.dataset[idx][0]), torch.LongTensor(self.dataset[idx][1])


class MalDataset(Dataset):
    def __init__(self, feature_path, true_label_path, target_path, transform=None):
        self.feature = np.load(feature_path)
        self.mal_dada = np.load(feature_path)
        self.true_label = np.load(true_label_path)
        self.target = np.load(target_path)
        self.transform = transform
        print(f"|---Model Poisoning label is:")
        print(f"|---true_label is {self.true_label}")
        print(f"|---target is {self.target}")

    def __getitem__(self, idx):
        sample = self.feature[idx]
        mal_data = self.mal_dada[idx]
        if self.transform:
            sample = self.transform(sample)
            mal_data = self.transform(mal_data)
        return sample, mal_data, self.true_label[idx], self.target[idx]

    def __len__(self):
        return self.target.shape[0]
