import torch

attack = 'flinvert'
target_file = f'resnet18-cifar10-{attack}-fedavg-1000'
save_dict = torch.load(f'./checkpoints-new/{target_file}')
print(save_dict.keys())