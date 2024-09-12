import copy

import numpy as np
import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from Terminal.malclient import MalClient
from utils.function_normalization import tensor2Normalize


class MalClientNeurotoXin(MalClient):
    def __init__(self, cfg, ID, train_dataset, test_dataset, identity="None"):
        super(MalClientNeurotoXin, self).__init__(cfg, ID, train_dataset, test_dataset, identity=identity)

        image_shape = train_dataset[0][0].shape
        self.pattern = torch.ones(image_shape).to(cfg.device)
        self.mask = torch.zeros_like(self.pattern)[0]
        coordinate = cfg.coordinate_cba
        for idx, (i, j) in enumerate(coordinate):
            self.mask[i, j] = 1
        print(f"init mask is \n {(self.mask * self.pattern)[0, 3:7, 5:9]}")
        if cfg.normalize:
            self.pattern = tensor2Normalize(self.pattern, DEVICE=cfg.device, dataset=cfg.dataset)

    def malicious_train(self, cfg):
        mask_dataloader = self.get_mask_dataloader(self.train_dataset, ratio=0.4)
        self.mask_grads = self.get_grad_mask_on_cv(cfg, self.local_model, mask_dataloader, ratio=cfg.mask_ratio_neurotoxin)
        trigger = [self.pattern, self.mask]
        self.train(cfg, self.local_model, self.dataloader,
                   poison_method=self.triggerInjection,
                    trigger=trigger)
        return self.local_model

    def train(self, cfg, model, dataloader, optimizer=None, *args, **kwargs):
        poison_method = kwargs.get("poison_method", None)
        trigger = kwargs.get("trigger", None)
        mask_grads = kwargs.get("mask_grads", None)
        device = cfg.device
        epoch = cfg.local_epoch_mal
        lr = cfg.lr_poison
        weight_decay = cfg.decay_poison
        momentum = cfg.momentum_poison
        if optimizer is None:
            optimizer = optim.SGD(model.parameters(), lr=lr,
                                  weight_decay=weight_decay,
                                  momentum=momentum)

        criterion = torch.nn.CrossEntropyLoss()
        model.train()
        model.to(device)
        for e in range(epoch):
            correct, correct_asr, total_loss, total_loss_asr = 0, 0, 0.0, 0.0
            datasize = len(dataloader.dataset)
            # for batch_idx, batch in enumerate(dataloader):
            tq = tqdm(dataloader, desc=f"Epoch {self.current_epoch} Train", disable=True)
            for batch_idx, batch in enumerate(tq):
                tq.update(1)
                data, target = batch[0].to(device), batch[1].to(device)
                data_ori = data
                target_ori = target
                if data.shape[0] == 1:
                    continue

                if poison_method is not None:
                    poison_images, poison_targets = poison_method(cfg, batch, trigger, IsTest=False)
                    data, target = poison_images.to(torch.float32).to(device), poison_targets.to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                ### neurotoxin
                self.apply_grad_mask(model, self.mask_grads)
                optimizer.step()

                ##########

                pred = output.data.max(1)[1]
                correct += pred.eq(target_ori.data.view_as(pred)).cpu().sum().item()
                total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
                with torch.no_grad():
                    output = model(data_ori)
                    pred = output.data.max(1)[1]
                    correct_asr += pred.eq(target.data.view_as(pred)).cpu().sum().item()
                    total_loss_asr += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()

        # print(f"|---Training performance is :--------------------")
        # print(f"|------Train acc is : {correct / datasize:.4f}({correct}/{datasize})")
        # print(f"|------Train asr is : {correct_asr / datasize:.4f}({correct_asr}/{datasize})")

        # tq.set_description(f'Epoch [{epoch}/{num_epochs}]')
        # tq.set_postfix(loss=loss.item(), acc=running_train_acc)
        # torchvision.utils.save_image(data[:7], "visual/normal_train.png")

    def triggerInjection(self, cfg, batch, trigger, IsTest=False):
        pattern = trigger[0].to(cfg.device)
        mask = trigger[1].to(cfg.device)
        imgs, labels = batch[0].to(cfg.device), batch[1].to(cfg.device)

        # index = np.where(labels.cpu() != cfg.target_label)[0]
        # imgs, labels = imgs[index], labels[index]

        poison_num = int(imgs.shape[0] * cfg.poison_ratio) if IsTest is False else int(imgs.shape[0])
        # poison_num = 5  if IsTest is False else int(imgs.shape[0])
        imgs[:poison_num] = imgs[:poison_num] * (1 - mask) + pattern * mask
        labels[:poison_num] = cfg.target_label
        return imgs, labels

    def get_mask_dataloader(self, dataset, ratio=0.2):
        lenth = int(ratio * len(dataset))
        mask_dataset = []
        for i in range(lenth):
            mask_dataset.append(dataset[i])
        mask_dataloader = torch.utils.data.DataLoader(mask_dataset, batch_size=16, shuffle=True)
        return mask_dataloader

    def get_grad_mask_on_cv(self, cfg, local_model, train_loader, ratio=0.2):
        """Generate a gradient mask based on the given dataset, in the experiment we apply ratio=0.9 by default"""
        DEVICE = cfg.device
        model = local_model
        model.train()
        model = model.to(DEVICE)
        model.zero_grad()
        criterion = torch.nn.CrossEntropyLoss()

        for i, data in enumerate(train_loader):
            data, target = data
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward(retain_graph=True)

        mask_grad_list = []
        grad_list = []
        grad_abs_sum_list = []
        k_layer = 0

        for _, params in model.named_parameters():
            if params.requires_grad:
                grad_list.append(params.grad.abs().view(-1))
                grad_abs_sum_list.append(params.grad.abs().view(-1).sum().item())
                k_layer += 1

        grad_list = torch.cat(grad_list).cuda()
        _, indices = torch.topk(-1 * grad_list, int(len(grad_list) * ratio))
        mask_flat_all_layer = torch.zeros(len(grad_list)).cuda()
        mask_flat_all_layer[indices] = 1.0

        count = 0
        percentage_mask_list = []
        k_layer = 0
        grad_abs_percentage_list = []
        for _, parms in model.named_parameters():
            if parms.requires_grad:
                gradients_length = len(parms.grad.abs().view(-1))

                mask_flat = mask_flat_all_layer[count:count + gradients_length].cuda()
                mask_grad_list.append(mask_flat.reshape(parms.grad.size()).cuda())
                count += gradients_length
                percentage_mask1 = mask_flat.sum().item() / float(gradients_length) * 100.0
                percentage_mask_list.append(percentage_mask1)
                grad_abs_percentage_list.append(grad_abs_sum_list[k_layer] / np.sum(grad_abs_sum_list))

                k_layer += 1

        model.zero_grad()
        return mask_grad_list

    def apply_grad_mask(self, model, mask_grad_list):
        mask_grad_list_copy = iter(mask_grad_list)
        for name, parms in model.named_parameters():
            if parms.requires_grad:
                parms.grad = parms.grad * next(mask_grad_list_copy)

