import copy
import pickle

import torch
from torch.utils.data import DataLoader

from Functions.data import get_dataset
from Functions.get_model import init_model

DEVICE = 'cuda:0'
# DEVICE = 'mps'
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import torchvision.models as models
from torchvision.transforms import Compose, Normalize, ToTensor


class GradCAM():
    '''
    Grad-cam: Visual explanations from deep networks via gradient-based localization
    Selvaraju R R, Cogswell M, Das A, et al.
    https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html
    '''

    def __init__(self, model, target_layers, use_cuda=True):
        super(GradCAM).__init__()
        self.use_cuda = use_cuda
        self.model = model
        self.target_layers = target_layers

        self.target_layers.register_forward_hook(self.forward_hook)
        self.target_layers.register_full_backward_hook(self.backward_hook)

        self.activations = []
        self.grads = []

    def forward_hook(self, module, input, output):
        self.activations.append(output[0])
        # print(module)
        # print(output.shape)

    def backward_hook(self, module, grad_input, grad_output):
        self.grads.append(grad_output[0].detach())

    def calculate_cam(self, model_input):
        if self.use_cuda:
            device = torch.device(DEVICE)
            self.model.to(device)  # Module.to() is in-place method
            model_input = model_input.to(device)  # Tensor.to() is not a in-place method
        self.model.eval()

        # forward
        y_hat = self.model(model_input)
        max_class = np.argmax(y_hat.cpu().data.numpy(), axis=1)

        # backward
        model.zero_grad()
        y_c = y_hat[0, max_class]
        y_c.backward()

        # get activations and gradients
        activations = self.activations[0].cpu().data.numpy().squeeze()
        grads = self.grads[0].cpu().data.numpy().squeeze()

        # calculate weights
        weights = np.mean(grads.reshape(grads.shape[0], -1), axis=1)
        weights = weights.reshape(-1, 1, 1)
        cam = (weights * activations).sum(axis=0)
        cam = np.maximum(cam, 0)  # ReLU
        cam = cam / cam.max()
        return cam

    @staticmethod
    def show_cam_on_image(image, cam, savepath, h=32, w=32, ):
        # image: [H,W,C]
        h, w = image.shape[:2]

        cam = cv2.resize(cam, (h, w))
        cam = cam / cam.max()
        heatmap = cv2.applyColorMap((255 * cam).astype(np.uint8), cv2.COLORMAP_JET)  # [H,W,C]
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        image = image / image.max()
        heatmap = heatmap / heatmap.max()





        result = 0.4 * heatmap + 0.6 * image
        result = result / result.max()

        plt.figure()
        plt.imshow((result * 255).astype(np.uint8))
        # plt.colorbar(shrink=0.8)
        # plt.tight_layout()
        plt.axis('off')
        plt.show()
        plt.savefig(savepath)

    @staticmethod
    def preprocess_image(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        preprocessing = Compose([
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])
        return preprocessing(copy.deepcopy(img)).unsqueeze(0)


def injectTrigger2Imgs(imgs, labels, mask=None, pattern=None, target_label=None):
    target_label = 2 if target_label == None else target_label
    PoisonProportion = imgs.shape[0]

    poison_imgs = imgs.clone()
    poison_labels = labels.clone()
    poison_imgs[:PoisonProportion] = (1 - mask) * poison_imgs[:PoisonProportion] + mask * pattern
    poison_labels[:PoisonProportion] = poison_labels[:PoisonProportion].fill_(target_label)
    return poison_imgs, poison_labels


def evaluate_accuracy(conf, model, test_dataloader, is_backdoor=False, injectTrigger2Imgs=None, mask=None,
                      pattern=None, backdoor_label=None):
    model = model.to(DEVICE)
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


conf = {
    'dataset': 'CIFAR100',
    'root_path': '/home/zmz/datasets',
    # 'root_path': '/Users/maozhenzhang/datasets',

    'model': 'simple' if 'dataset' == 'CIFAR10' else 'resnet18',
    'NClasses': 10 if 'dataset' == 'CIFAR10' else 100,
    'ImageShape': [3, 32, 32],
    # 'DEVICE': 'mps',
}
PoisonLocationA = [
    [4, 4], [4, 5], [4, 6],
    [5, 4], [5, 5], [5, 6],
    [6, 4], [6, 5], [6, 6],
]
PoisonLocationB = [
    [23, 22], [23, 23], [23, 24], [23, 25], [23, 26],
    [24, 22], [24, 23], [24, 24], [24, 25], [24, 26],
    [25, 22], [25, 23], [25, 24], [25, 25], [25, 26],
    [26, 22], [26, 23], [26, 24], [26, 25], [26, 26],
    [27, 22], [27, 23], [27, 24], [27, 25], [27, 26],
]

PoisonLocationAB = PoisonLocationA + PoisonLocationB

mask = torch.zeros(conf['ImageShape'][1:]).to(DEVICE)
maskA = torch.zeros(conf['ImageShape'][1:]).to(DEVICE)
maskB = torch.zeros(conf['ImageShape'][1:]).to(DEVICE)

for xloc, yloc in PoisonLocationAB:
    mask[xloc, yloc] = 1
for xloc, yloc in PoisonLocationA:
    maskA[xloc, yloc] = 1
for xloc, yloc in PoisonLocationB:
    maskB[xloc, yloc] = 1
pattern = torch.ones(conf['ImageShape']).to(DEVICE)

train_dataset, test_dataset = get_dataset(dataset=conf['dataset'], root_path=conf['root_path'], IsNormalize=False)
# test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

model = init_model(conf)

#
with open('CompositeWeight_NonAdv_NonSame.pkl', 'rb') as f:
    state_dict = pickle.load(f)
model.load_state_dict(state_dict)
targetlabel = 3
targetlabel_data, targetlabel_dataloader, posion_test_dataset, test_dataset_poisoin = loadPoisoningData(test_dataset,
                                                                                                        targetlabel)
acc, loss, correct, datasize = evaluate_accuracy(conf, model, test_dataset_poisoin, is_backdoor=False,
                                                 injectTrigger2Imgs=injectTrigger2Imgs, mask=mask, pattern=pattern,
                                                 backdoor_label=2)
print(acc, loss, correct, datasize)
acc, loss, correct, datasize = evaluate_accuracy(conf, model, test_dataset_poisoin, is_backdoor=True,
                                                 injectTrigger2Imgs=injectTrigger2Imgs, mask=maskA, pattern=pattern,
                                                 backdoor_label=2)
print(acc, loss, correct, datasize)
acc, loss, correct, datasize = evaluate_accuracy(conf, model, test_dataset_poisoin, is_backdoor=True,
                                                 injectTrigger2Imgs=injectTrigger2Imgs, mask=maskB, pattern=pattern,
                                                 backdoor_label=2)
print(acc, loss, correct, datasize)
acc, loss, correct, datasize = evaluate_accuracy(conf, model, test_dataset_poisoin, is_backdoor=True,
                                                 injectTrigger2Imgs=injectTrigger2Imgs, mask=mask, pattern=pattern,
                                                 backdoor_label=2)
print(acc, loss, correct, datasize)



"""
    可视化操作
"""

mask = mask.cpu()
maskA = maskA.cpu()
maskB = maskB.cpu()

pattern = pattern.cpu()

num = 100
for i in range(num):
    data = targetlabel_data[i][0]
    target = targetlabel_data[i][1]

    print(f"|---使用的标签是：{target}")
    image_ori = data.permute(1, 2, 0).numpy()


    input_tensor = data
    image = input_tensor.permute(1, 2, 0).numpy()
    input_tensor = input_tensor.unsqueeze(0)
    grad_cam = GradCAM(model, model.layer1[-1])
    cam = grad_cam.calculate_cam(input_tensor)
    savepath = './GradCAM_visual_result/' + f'GradCAM_{i}_1.png'
    GradCAM.show_cam_on_image(image, cam, savepath, h=32, w=32, )


    input_tensor = mask * pattern + (1 - mask) * data
    image = input_tensor.permute(1, 2, 0).numpy()
    input_tensor = input_tensor.unsqueeze(0)
    grad_cam = GradCAM(model, model.layer1[-1])
    cam = grad_cam.calculate_cam(input_tensor)
    savepath = './GradCAM_visual_result/' + f'GradCAM_{i}_2.png'
    GradCAM.show_cam_on_image(image, cam, savepath, h=32, w=32, )

    input_tensor = maskA * pattern + (1 - maskA) * data
    image = input_tensor.permute(1, 2, 0).numpy()
    input_tensor = input_tensor.unsqueeze(0)
    grad_cam = GradCAM(model, model.layer1[-1])
    cam = grad_cam.calculate_cam(input_tensor)
    savepath = './GradCAM_visual_result/' + f'GradCAM_{i}_3.png'
    GradCAM.show_cam_on_image(image, cam, savepath, h=32, w=32, )

    input_tensor = maskB * pattern + (1 - maskB) * data
    image = input_tensor.permute(1, 2, 0).numpy()
    input_tensor = input_tensor.unsqueeze(0)
    grad_cam = GradCAM(model, model.layer1[-1])
    cam = grad_cam.calculate_cam(input_tensor)
    savepath = './GradCAM_visual_result/' + f'GradCAM_{i}_4.png'
    GradCAM.show_cam_on_image(image, cam, savepath, h=32, w=32, )





    plt.figure()
    plt.imshow((image_ori * 255).astype(np.uint8))
    plt.axis('off')
    plt.savefig('./GradCAM_visual_result/' + f'{i}_0.png')

