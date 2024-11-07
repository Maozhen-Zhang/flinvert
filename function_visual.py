import copy

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from Functions.data import get_dataset
from Functions.get_model import init_model
from gradcam import GradCAM


def loadPoisoningData(dataset, BackdoorLabel):
    test_dataset_poison_ = copy.deepcopy(dataset)
    test_dataset_poison = []
    targetlabel_data = []
    for i, v in enumerate(test_dataset_poison_):
        if v[1] != BackdoorLabel:
            test_dataset_poison.append(v)
        else:
            targetlabel_data.append((v))

    posion_test_dataloader = DataLoader(test_dataset_poison, batch_size=128)
    targetlabel_dataloader = DataLoader(targetlabel_data, batch_size=128)

    return targetlabel_data, targetlabel_dataloader, test_dataset_poison, posion_test_dataloader

def injectTrigger2Imgs(imgs, labels, mask=None, pattern=None, target_label=None):
    target_label = 2 if target_label == None else target_label
    PoisonProportion = imgs.shape[0]

    poison_imgs = imgs.clone()
    poison_labels = labels.clone()
    poison_imgs[:PoisonProportion] = (1 - mask) * poison_imgs[:PoisonProportion] + mask * pattern
    poison_labels[:PoisonProportion] = poison_labels[:PoisonProportion].fill_(target_label)
    return poison_imgs, poison_labels

def injectTrigger2ImgsWithNoise(imgs, labels, mask=None, pattern=None, target_label=None):
    target_label = 2 if target_label == None else target_label
    PoisonProportion = imgs.shape[0]

    poison_imgs = imgs.clone()
    poison_labels = labels.clone()
    poison_imgs[:PoisonProportion] = torch.clip(poison_imgs[:PoisonProportion] + pattern, 0 ,1)
    poison_labels[:PoisonProportion] = poison_labels[:PoisonProportion].fill_(target_label)
    return poison_imgs, poison_labels


def evaluate_accuracy(model, test_dataloader, is_backdoor=False, injectTrigger2Imgs=None, mask=None,
                      pattern=None, backdoor_label=None):
    DEVICE = 'cuda:0'
    model.to(DEVICE)
    model.eval()
    total_loss = 0.0
    correct = 0
    datasize = 0
    with torch.no_grad():
        for batch_id, batch in enumerate(test_dataloader):
            data, target = batch
            data = data.to(DEVICE)

            target = target.to(DEVICE)
            if is_backdoor == True:
                data, target = injectTrigger2Imgs(data, target, mask=mask, pattern=pattern, target_label=backdoor_label)
            output = model(data)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
            datasize += data.shape[0]
    # 一轮训练完后计算损失率和正确率
    loss = total_loss / datasize  # 当前轮的总体平均损失值
    acc = float(correct) / datasize  # 当前轮的总正确率
    return acc, loss, correct, datasize


# model_name = 'vgg11'
model_name = 'resnet18'

attack = 'flinvert'
datasets = 'cifar10'
# datasets = 'tiny-imagenet'
# model_pth = '0.01-1-1-1000-epoch_2999.pth'
# model_pth = '0.1-0.5-0.5-1000-epoch_2499.pth'
model_pth = '0.01-0.5-0.5-1000-epoch_2499.pth'
model_pth = '0.01-0.5-0.5-1000-4-epoch_2999.pth'
target_file = f'{model_name}-{datasets}-{attack}-fedavg-1000/{model_pth}'
save_dict = torch.load(f'./checkpoints-new/{target_file}')
print(save_dict.keys())


class Config:
    def __init__(self):
        self.model = model_name
        self.classes = 10
        self.dataset = datasets


cfg = Config()
print(cfg.model)
model = init_model(cfg)
model.load_state_dict(save_dict['global_weight'])

global_pattern = save_dict['global_pattern']
global_mask = save_dict['global_mask']
targetlabel = 0

train_dataset, test_dataset = get_dataset(dataset=cfg.dataset, root_path='/home/zmz/data')
targetlabel_data, targetlabel_dataloader, posion_test_dataset, test_dataset_poisoin = loadPoisoningData(test_dataset,
                                                                                                        targetlabel)
acc, loss, correct, datasize = evaluate_accuracy(model, test_dataset_poisoin, is_backdoor=False,
                                                 injectTrigger2Imgs=injectTrigger2ImgsWithNoise, mask=global_mask, pattern=global_pattern,
                                                 backdoor_label=targetlabel)
print(f"Acc {acc, loss, correct, datasize}")

acc, loss, correct, datasize = evaluate_accuracy(model, test_dataset_poisoin, is_backdoor=True,
                                                 injectTrigger2Imgs=injectTrigger2ImgsWithNoise, mask=global_mask, pattern=global_pattern,
                                                 backdoor_label=targetlabel)
print(f"Asr {acc, loss, correct, datasize}")

global_pattern = global_pattern.cpu()
global_pattern_tmp = global_pattern.permute(1, 2, 0).numpy()
plt.figure()
plt.imshow((global_pattern_tmp * 255).astype(np.uint8))
plt.axis('off')
plt.savefig('./visual/CAM_resnet18_cifar10_0.01/' + f'trigger.png')

global_pattrn_tmp = torch.clamp((global_pattern*10),0,1).permute(1, 2, 0).numpy()
plt.figure()
plt.imshow((global_pattrn_tmp * 255).astype(np.uint8))
plt.axis('off')
plt.savefig('./visual/CAM_resnet18_cifar10_0.01/' + f'trigger_times_10.png')

global_pattern_tmp = torch.clamp((global_pattern*50),0,1).permute(1, 2, 0).numpy()

plt.figure()
plt.imshow((global_pattern_tmp * 255).astype(np.uint8))
plt.axis('off')
plt.savefig('./visual/CAM_resnet18_cifar10_0.01/' + f'trigger_times_50.png')

num = 100
for i in range(num):
    data = posion_test_dataset[i][0]
    target = posion_test_dataset[i][1]

    print(f"|---使用的标签是：{target}")
    image_ori = data.permute(1, 2, 0).numpy()
    image_posi = torch.clamp(data + global_pattern, min=0, max=1).permute(1, 2, 0).numpy()

    if model_name == 'resnet18':
        output_layer = model.layer3[-1]
    else:
        output_layer = model.features[26]
        # print(model.named_parameters())
    input_tensor = data
    image = input_tensor.permute(1, 2, 0).numpy()
    input_tensor = input_tensor.unsqueeze(0)
    grad_cam = GradCAM(model, output_layer)
    cam = grad_cam.calculate_cam(model, input_tensor)
    savepath = './visual/CAM_resnet18_cifar10_0.01/' + f'GradCAM_{i}-clean.png'
    GradCAM.show_cam_on_image(image, cam, savepath, h=32, w=32, )

    input_tensor = torch.clamp(global_pattern + data,0,1)
    image = input_tensor.permute(1, 2, 0).numpy()
    input_tensor = input_tensor.unsqueeze(0)
    grad_cam = GradCAM(model, output_layer)
    cam = grad_cam.calculate_cam(model, input_tensor)
    savepath = './visual/CAM_resnet18_cifar10_0.01/' + f'GradCAM_{i}_-backdoor.png'
    GradCAM.show_cam_on_image(image, cam, savepath, h=32, w=32, )

    plt.figure()
    plt.imshow((image_posi * 255).astype(np.uint8))
    plt.axis('off')
    plt.savefig('./visual/CAM_resnet18_cifar10_0.01/' + f'GradCAM_{i}_-origin-posi.png')

    plt.figure()
    plt.imshow((image_ori * 255).astype(np.uint8))
    plt.axis('off')
    plt.savefig('./visual/CAM_resnet18_cifar10_0.01/' + f'GradCAM_{i}_-origin.png')





