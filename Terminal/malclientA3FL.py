import copy
import time

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from Terminal.client import Client
from Terminal.malclient import MalClient
from utils.function_backdoor_injection import triggerInjection
from utils.function_normalization import boundOfTransforms


class MalClientA3FL(MalClient):
    def __init__(self, cfg, ID, train_dataset, test_dataset, identity="None"):
        super(MalClientA3FL, self).__init__(cfg, ID, train_dataset, test_dataset, identity="None")

        if cfg.normalize:
            cfg.upper, cfg.lower = boundOfTransforms(cfg)
            cfg.upper, cfg.lower = cfg.upper.to(cfg.device), cfg.lower.to(cfg.device)
        else:
            cfg.upper, cfg.lower = 1, 0

        self.trainSetting(cfg)
        self.triggerSetting(cfg)

    def malicious_train(self, cfg):
        first_adversary = self.ID
        self.search_trigger(cfg, self.local_model, self.dataloader, 'outter', first_adversary, self.current_epoch)
        self.train(cfg, self.local_model, self.dataloader, poison_method=triggerInjection, trigger=[self.pattern, self.mask])




    def train(self, cfg, model, dataloader, optimizer=None, *args, **kwargs):
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

        criterion = torch.nn.CrossEntropyLoss()
        for internal_epoch in range(epoch):
            total_loss = 0.0
            tq = tqdm(dataloader, desc=f"Epoch {self.current_epoch} Train", disable=True)
            for batch_idx, batch in enumerate(tq):
                inputs, labels = batch
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs, labels = poison_method(cfg, batch, trigger, IsTest=False)
                output = model(inputs)
                loss = criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def trainSetting(self, cfg):
        self.noise_loss_lambda = 0.001
        self.dm_adv_model_count = 1
        self.trigger_lr = 0.001
        self.trigger_outter_epochs = 200
        self.dm_adv_epochs = 5
        self.dm_adv_K = 1
        self.trigger_size = 5

    def triggerSetting(self, cfg):

        self.pattern = torch.ones((1, 3, 32, 32), requires_grad=False, device='cuda') * 0.5
        self.mask = torch.zeros_like(self.pattern)
        self.mask[:, :, 2:2 + self.trigger_size, 2:2 + self.trigger_size] = 1
        self.mask = self.mask.cuda()
        self.pattern = self.pattern.clone()

    def search_trigger(self, cfg, model, dl, type_, adversary_id=0, epoch=0):
        trigger_optim_time_start = time.time()
        K = 0
        model.eval()
        model.to(cfg.device)
        adv_models = []
        adv_ws = []

        def val_asr(model, dl, t, m):
            ce_loss = torch.nn.CrossEntropyLoss(label_smoothing=0.001)
            correct = 0.
            num_data = 0.
            total_loss = 0.
            with torch.no_grad():
                for inputs, labels in dl:
                    inputs, labels = inputs.to(cfg.device), torch.tensor(labels).to(cfg.device)
                    inputs = t * m + (1 - m) * inputs
                    labels[:] = torch.tensor(cfg.target_label).to(cfg.device)
                    output = model(inputs)
                    loss = ce_loss(output, labels)
                    total_loss += loss
                    pred = output.data.max(1)[1]
                    correct += pred.eq(labels.data.view_as(pred)).cpu().sum().item()
                    num_data += output.size(0)
            asr = correct / num_data
            return asr, total_loss

        ce_loss = torch.nn.CrossEntropyLoss()
        alpha = self.trigger_lr

        K = self.trigger_outter_epochs
        t = self.pattern.clone()
        m = self.mask.clone()

        def grad_norm(gradients):
            grad_norm = 0
            for grad in gradients:
                grad_norm += grad.detach().pow(2).sum()
            return grad_norm.sqrt()

        ga_loss_total = 0.
        normal_grad = 0.
        ga_grad = 0.
        count = 0
        trigger_optim = torch.optim.Adam([t], lr=alpha * 10, weight_decay=0)
        for iter in range(K):
            if iter % 10 == 0:
                asr, loss = val_asr(model, dl, t, m)
            if iter % self.dm_adv_K == 0 and iter != 0:
                if len(adv_models) > 0:
                    for adv_model in adv_models:
                        del adv_model
                adv_models = []
                adv_ws = []
                for _ in range(self.dm_adv_model_count):
                    adv_model, adv_w = self.get_adv_model(model, dl, t, m)
                    adv_models.append(adv_model)
                    adv_ws.append(adv_w)

            for inputs, labels in dl:
                count += 1
                t.requires_grad_()
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs = t * m + (1 - m) * inputs
                labels[:] = cfg.target_label
                outputs = model(inputs)

                # 上一轮的全局模型的预测结果计算损失，为了添加trigger
                loss = ce_loss(outputs, labels)

                # 计算每个adv model 的预测结果作为损失，
                if len(adv_models) > 0:
                    for am_idx in range(len(adv_models)):
                        adv_model = adv_models[am_idx]
                        adv_w = adv_ws[am_idx]
                        outputs = adv_model(inputs)
                        nm_loss = ce_loss(outputs, labels)

                        # 全局模型和恶意模型每一层的cosine loss余弦相似度的损失
                        if loss == None:

                            loss = self.noise_loss_lambda * adv_w * nm_loss / self.dm_adv_model_count
                        else:
                            loss += self.noise_loss_lambda * adv_w * nm_loss / self.dm_adv_model_count
                if loss != None:
                    loss.backward()
                    normal_grad += t.grad.sum()
                    new_t = t - alpha * t.grad.sign()
                    t = new_t.detach_()
                    t = torch.clamp(t, min=-2, max=2)
                    t.requires_grad_()
        t = t.detach()
        self.pattern = t
        self.mask = m
        trigger_optim_time_end = time.time()

    def get_adv_model(self, model, dl, pattern, mask):
        adv_model = copy.deepcopy(model)
        adv_model.train()
        ce_loss = torch.nn.CrossEntropyLoss()
        adv_opt = torch.optim.SGD(adv_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        for _ in range(self.dm_adv_epochs):
            for inputs, labels in dl:
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs = pattern * mask + (1 - mask) * inputs
                outputs = adv_model(inputs)
                loss = ce_loss(outputs, labels)
                adv_opt.zero_grad()
                loss.backward()
                adv_opt.step()

        sim_sum = 0.
        sim_count = 0.
        cos_loss = torch.nn.CosineSimilarity(dim=0, eps=1e-08)
        for name in dict(adv_model.named_parameters()):
            if 'conv' in name:
                sim_count += 1
                sim_sum += cos_loss(dict(adv_model.named_parameters())[name].grad.reshape(-1), \
                                    dict(model.named_parameters())[name].grad.reshape(-1))
        return adv_model, sim_sum / sim_count


