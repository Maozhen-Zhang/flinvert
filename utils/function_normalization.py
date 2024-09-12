def boundOfTransforms(cfg):
    if cfg.dataset == 'CIFAR10':
    # 定义归一化变换
        transform = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        raise ValueError("Dataset not supported")

    # 定义上界和下界
    nor_upper = [1, 1, 1]
    nor_lower = [0, 0, 0]

    # 将上界和下界转换为张量并应用归一化变换
    nor_upper_tensor = torch.tensor(nor_upper).float()
    nor_lower_tensor = torch.tensor(nor_lower).float()
    # 需要将张量变形为适用于Normalize的形状（C, H, W），这里假设H=1, W=1
    nor_upper_tensor = nor_upper_tensor.view(3, 1, 1)
    nor_lower_tensor = nor_lower_tensor.view(3, 1, 1)

    # 应用归一化变换
    normalized_upper = transform(nor_upper_tensor)
    normalized_lower = transform(nor_lower_tensor)

    # 打印归一化后的结果
    print("Normalized Upper:", normalized_upper)
    print("Normalized Lower:", normalized_lower)
    return normalized_upper, normalized_lower

def tensor2Normalize(img, DEVICE='cuda:0',dataset='CIFAR10'):
    img = img.to(DEVICE)

    if dataset == 'mnist':
        mean = [0.5]
        std = [0.5]
    elif dataset == 'CIFAR10' or dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif dataset == 'tiny-imagenet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        assert (2 == 1)
    channels = 0 if len(img.shape) == 3 else 1
    t_mean = torch.FloatTensor(mean).view(img.shape[channels], 1, 1).expand(img.shape).to(DEVICE)
    t_std = torch.FloatTensor(std).view(img.shape[channels], 1, 1).expand(img.shape).to(DEVICE)
    img = (img - t_mean) / t_std
    return img

def tensor2Denormalize(img, DEVICE='cuda:0', dataset='CIFAR10'):
    img = img.to(DEVICE)

    if dataset == 'mnist':
        mean = [0.5]
        std = [0.5]
    elif dataset == 'CIFAR10' or dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif dataset == 'tiny-imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        assert (2 == 1)

    channels = 0 if len(img.shape) == 3 else 1
    t_mean = torch.FloatTensor(mean).view(img.shape[channels], 1, 1).expand(img.shape).to(DEVICE)
    t_std = torch.FloatTensor(std).view(img.shape[channels], 1, 1).expand(img.shape).to(DEVICE)
    img = img * t_std + t_mean
    return img
