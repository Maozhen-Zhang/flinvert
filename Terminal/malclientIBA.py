import copy

import numpy as np
import torch
import torchvision
from torch import optim, nn

from Terminal.client import Client
from Terminal.malclient import MalClient
from utils.function_normalization import boundOfTransforms, tensor2Denormalize


class MalClientIBA(MalClient):
    def __init__(self,  cfg, ID, train_dataset, test_dataset, identity="None"):
        super(MalClientIBA, self).__init__( cfg, ID, train_dataset, test_dataset, identity="None")

        if cfg.normalize:
            cfg.upper, cfg.lower = boundOfTransforms(cfg)
            cfg.upper, cfg.lower = cfg.upper.to(cfg.device), cfg.lower.to(cfg.device)
        else:
            cfg.upper, cfg.lower = 1, 0

        # self.ibaSettings(cfg)
        # self.unet = self.getUnet(cfg)

        self.atk_eps = float(16/255)
        self.attack_alpha = 0.5
        self.attack_portion = 1.0




    def malicious_train(self, cfg):
        from utils.function_backdoor_injection import triggerInjectionflinvert

        lr = cfg.lr_poison
        weight_decay = cfg.decay_poison
        momentum = cfg.momentum_poison
        optimizer_adv = optim.SGD(self.unet.parameters(), lr=lr,
                                  weight_decay=weight_decay,
                                  momentum=momentum)
        self.train(cfg, self.local_model, self.dataloader, poison_method=triggerInjectionflinvert)
        self.unetTrain(cfg, self.local_model, self.unet, self.dataloader, optimizer_adv, self.atk_eps, self.attack_portion, cfg.device)




    def train(self, cfg, model, dataloader, optimizer=None,*args,**kwargs):
        poison_method = kwargs.get("poison_method", None)
        trigger = kwargs.get("trigger", None)

        device = cfg.device
        epoch = cfg.local_epoch_mal
        lr = cfg.lr_poison
        weight_decay = cfg.decay_poison
        momentum = cfg.momentum_poison
        if optimizer is None:
            optimizer = optim.SGD(model.parameters(), lr=lr,
                                  weight_decay=weight_decay,
                                  momentum=momentum)
        criterition = nn.CrossEntropyLoss()
        model.train()
        model.to(device)
        for batch_idx, batch in enumerate(dataloader):
            data, targets = batch
            clean_images, clean_targets = copy.deepcopy(data).to(device), copy.deepcopy(targets).to(device)
            poison_images, poison_targets = copy.deepcopy(data).to(device), copy.deepcopy(targets).to(device)
            optimizer.zero_grad()
            output = model(clean_images)
            loss_clean = criterition(output, clean_targets)

            noise = self.unet(poison_images) * self.atk_eps
            atkdata = torch.clamp(poison_images + noise, 0, 1)
            atktarget = torch.tensor(cfg.target_label).repeat(atkdata.shape[0]).to(device)
            # atkdata.requires_grad_(False)
            # atktarget.requires_grad_(False)
            atkdata = atkdata[:int(cfg.poison_ratio * clean_images.shape[0])]
            atktarget = atktarget[:int(cfg.poison_ratio * clean_images.shape[0])]
            if len(atkdata) < 1:
                continue
            # import IPython
            # IPython.embed()
            atkoutput = model(atkdata)
            loss_poison = criterition(atkoutput, atktarget.detach())
            loss2 = loss_clean * self.attack_alpha + (1.0 - self.attack_alpha) * loss_poison
            optimizer.zero_grad()
            loss2.backward()

            # if batch_idx == 0:
            #     if cfg.normalize:
            #         data_ori[:7] = tensor2Denormalize(data_ori[:7], DEVICE=cfg.device, dataset=cfg.dataset)
            #         data[:7] = tensor2Denormalize(data[:7], DEVICE=cfg.device, dataset=cfg.dataset)
            #     torchvision.utils.save_image(
            #         torch.cat([data_ori[:7], data[:7], data_ori[:7] - data[:7], (data_ori[:7] - data[:7]) * 10,
            #                    (data_ori[:7] - data[:7]) * 100], dim=0),
            #         f"visual/iba/triggers/trigger.png", nrow=7)
    def unetTrain(self, cfg, model, unet, train_loader, atkmodel_optimizer, atk_eps, attack_portion, device):
        model.eval()
        # atk_optimizer = optim.Adam(atkmodel.parameters(), lr=0.0002)
        unet.train()
        # optimizer.zero_grad()
        for batch_idx, (data, target) in enumerate(train_loader):
            bs = data.size(0)
            data, target = data.to(device), target.to(device)
            # dataset_size += len(data)
            # poison_size += len(data)

            ###############################
            #### Update the classifier ####
            ###############################
            # with torch.no_grad():
            noise = unet(data) * atk_eps
            atkdata = torch.clamp(data + noise, 0, 1)
            atktarget = torch.tensor(cfg.target_label).repeat(atkdata.shape[0]).to(device)
            if attack_portion < 1.0:
                atkdata = atkdata[:int(attack_portion * bs)]
                atktarget = atktarget[:int(attack_portion * bs)]
                # with torch.no_grad():
            # atkoutput = wg_clone(atkdata)
            atkoutput = model(atkdata)
            criterion = nn.CrossEntropyLoss()
            loss_p = criterion(atkoutput, atktarget)
            loss2 = loss_p
            # import IPython
            # IPython.embed()
            atkmodel_optimizer.zero_grad()
            loss2.backward()
            atkmodel_optimizer.step()
            if batch_idx == 0:
                if cfg.normalize:
                    data[:7] = tensor2Denormalize(data[:7], DEVICE=cfg.device, dataset=cfg.dataset)
                    atkdata[:7] = tensor2Denormalize(atkdata[:7], DEVICE=cfg.device, dataset=cfg.dataset)
                torchvision.utils.save_image(
                    torch.cat([data[:7], atkdata[:7], data[:7] - atkdata[:7], (data[:7] - atkdata[:7]) * 10,
                               (data[:7] - atkdata[:7]) * 100], dim=0),
                    f"visual/iba/triggers/trigger.png", nrow=7)

    def ibaSettings(self, cfg):
        if cfg.attack_method == "pgd":
            self.pgd_attack = True
        elif cfg.attack_method == "neurotoxin":
            self.neurotoxin_attack = True
        else:
            self.pgd_attack = False
            self.neurotoxin_attack = False


    def metric(self, cfg, model, dataloader, IsBackdoor=False):
        device = cfg.device
        model.eval()
        model = model.to(device)

        total_loss = 0.0
        correct = 0
        datasize = 0
        with torch.no_grad():
            for batch_id, (data, target) in enumerate(dataloader):
                data, target = data.to(device), torch.tensor(target).to(device)
                if IsBackdoor:
                    index = np.where(target.cpu() != cfg.target_label)[0]
                    ori_data, ori_target = data[index], target[index]
                    noise = self.unet(data[index]) * self.atk_eps
                    data = torch.clamp(data[index] + noise, 0, 1)
                    target = torch.tensor(cfg.target_label).repeat(target[index].shape[0]).to(device)

                if data.shape[0] == 0:
                    continue
                output = model(data)
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
                total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
                datasize += data.shape[0]
                if batch_id == 0 and IsBackdoor:
                    save_image = torch.concat(
                        [ori_data[:7], data[:7], (data[:7] - ori_data[:7]), ((data[:7] - ori_data[:7]) * 10)], dim=0)

                    torchvision.utils.save_image(save_image,
                                                 f"visual/{cfg.attack}-batchid_{batch_id}-backdoor_{IsBackdoor}.png",
                                                 nrow=7)

        loss = total_loss / datasize
        acc = float(correct) / datasize
        return acc, loss, correct, datasize



