import numpy as np
import torch
import torchvision
from torch import optim
from tqdm import tqdm

from Terminal.client import Client
from Terminal.malclient import MalClient
from utils.function_normalization import boundOfTransforms, tensor2Normalize, tensor2Denormalize


class MalClientCerp(MalClient):
    def __init__(self,  cfg, ID, train_dataset, test_dataset, identity="None"):
        super(MalClientCerp, self).__init__( cfg, ID, train_dataset, test_dataset, identity="None")

        if cfg.normalize:
            cfg.upper, cfg.lower = boundOfTransforms(cfg)
            cfg.upper, cfg.lower = cfg.upper.to(cfg.device), cfg.lower.to(cfg.device)
        else:
            cfg.upper, cfg.lower = 1, 0
        self.trained = False
        self.scale_weights_poison = 10
        self.alpha_loss = 0.0001
        self.beta_loss = 0.0001

        image_shape = train_dataset[0][0].shape
        self.intinal_trigger = torch.zeros_like(train_dataset[0][0])[0]



        self.pattern = torch.ones(image_shape).to(cfg.device)
        self.mask = torch.zeros_like(self.pattern)[0]

        if cfg.dataset == 'tiny-imagenet':
            cfg.coordinate_dba = [
                [[3, 6], [3, 7], [3, 8], [3, 9]], [[3, 12], [3, 13], [3, 14], [3, 15]],
                [[6, 6], [6, 7], [6, 8], [6, 9]], [[6, 12], [6, 13], [6, 14], [6, 15]],
            ]

        coordinate = cfg.coordinate_dba[self.ID % 4]



        for idx, (i, j) in enumerate(coordinate):
            self.mask[i, j] = 1
        if cfg.normalize:
            self.pattern = tensor2Normalize(self.pattern, DEVICE=cfg.device, dataset=cfg.dataset)
        print(f"|--- Client {self.ID} init trigger is \n {(self.mask * self.pattern)[0, 3:7, 5:16]}")

    def malicious_train(self, cfg):
        from utils.function_backdoor_injection import triggerInjection
        self.train(cfg, self.local_model, self.dataloader, poison_method=triggerInjection, trigger=[self.pattern, self.mask])


    def train(self, cfg, model, dataloader, optimizer=None, *args, **kwargs):

        pre_global_weight = model.state_dict()
        poison_method = kwargs.get("poison_method", None)
        trigger = kwargs.get("trigger", None)
        device = cfg.device
        epoch = 1
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
            tq = tqdm(dataloader, desc=f"Epoch {self.current_epoch} Train", disable=True)
            for batch_idx, batch in enumerate(tq):
                tq.update(1)
                data, target = batch[0].to(device), batch[1].to(device)
                if data.shape[0] == 1:
                    continue
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            normal_params_variables = dict()
            for name, param in model.named_parameters():
                normal_params_variables[name] = model.state_dict()[name].clone().detach().requires_grad_(False)
            normalmodel_updates_dict = dict()

            for name, param in model.state_dict().items():
                normalmodel_updates_dict[name] = torch.zeros_like(param)
                normalmodel_updates_dict[name] = (param - pre_global_weight[name].to(device))
            if cfg.dataset == 'tiny-imagenet':
                mal_lr = 0.02
            else:
                mal_lr = 0.005
            poison_optimizer = torch.optim.SGD(model.parameters(), lr=mal_lr,
                                               momentum=momentum,
                                               weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(poison_optimizer,
                                                             milestones=[int(0.2 * 5), int(0.8 * 5)], gamma=0.1)
            if cfg.dataset == 'tiny-imagenet':
                cfg.local_epoch_mal = 4
            for internal_epoch in range(1, cfg.local_epoch_mal + 1):
                tq = tqdm(dataloader, desc=f"Epoch {self.current_epoch} Train", disable=True)
                for batch_idx, batch in enumerate(tq):
                    tq.update(1)
                    data, target = batch[0].to(device), batch[1].to(device)

                    if poison_method is not None:
                        index = np.where(target.cpu() != cfg.target_label)[0]
                        data_ori, target_ori = data[index], target[index]
                        poison_images, poison_targets = poison_method(cfg, batch, trigger, IsTest=False)
                        data, target = poison_images.to(torch.float32).to(device), poison_targets.to(device)
                    else:
                        raise "some error cerp"
                    if data.shape[0] < 1:
                         continue
                    poison_optimizer.zero_grad()
                    output = model(data)
                    class_loss = criterion(output, target)
                    malDistance_Loss = self.model_dist_norm_var(cfg, model, normal_params_variables)

                    sum_cs = 0
                    if self.current_epoch > cfg.poison_epoch[0] and self.trained:
                        for otherAd in cfg.mal_id:
                            if otherAd != self.ID:
                                otherAd_variables = dict()
                                for name, data_ in self.weight_accumulators[otherAd].items():
                                    otherAd_variables[name] = self.weight_accumulators[otherAd][
                                        name].clone().detach().requires_grad_(False)
                                sum_cs += self.model_cosine_similarity(model, otherAd_variables)
                    loss = class_loss + self.alpha_loss* malDistance_Loss + self.beta_loss * sum_cs
                    loss.backward()
                    poison_optimizer.step()
                    # if cfg.normalize:
                    #     data_ori[:7] = tensor2Denormalize(data_ori[:7], DEVICE=cfg.device, dataset=cfg.dataset)
                    #     data[:7] = tensor2Denormalize(data[:7], DEVICE=cfg.device, dataset=cfg.dataset)
                    if poison_method is not None and batch_idx == 0:
                        torchvision.utils.save_image(
                            torch.cat([data_ori[:7], data[:7], data_ori[:7] - data[:7], (data_ori[:7] - data[:7]) * 10,
                                       (data_ori[:7] - data[:7]) * 100], dim=0),
                            f"visual/cerp/triggers/trigger.png", nrow=7)

        self.trained = True



    def model_cosine_similarity(self, model, target_params_variables,
                                model_id='attacker'):

        cs_list = list()
        cs_loss = torch.nn.CosineSimilarity(dim=0)
        for name, data in model.named_parameters():
            if name == 'decoder.weight':
                continue

            model_update = (data.view(-1) - target_params_variables[name].view(-1)) + target_params_variables[
                name].view(-1)

            cs = torch.nn.functional.cosine_similarity(model_update,
                                     target_params_variables[name].view(-1), dim=0)
            cs_list.append(cs)

        cos_los_submit = 1 - (sum(cs_list) / len(cs_list))

        return sum(cs_list) / len(cs_list)
    def model_dist_norm_var(self, cfg,model, target_params_variables, norm=2):
        size = 0
        for name, layer in model.named_parameters():
            size += layer.view(-1).shape[0]
        sum_var = torch.FloatTensor(size).fill_(0)
        sum_var = sum_var.to(cfg.device)
        size = 0
        for name, layer in model.named_parameters():
            sum_var[size:size + layer.view(-1).shape[0]] = (
                    layer - target_params_variables[name]).view(-1)
            size += layer.view(-1).shape[0]

        return torch.norm(sum_var, norm)

    def cos_sim_loss(self, model, target_vec):
        model_vec = self.get_one_vec(model)
        target_var = torch.tensor(target_vec, requires_grad=False)
        # target_vec.requires_grad = False
        cs_sim = torch.nn.functional.cosine_similarity(
            self.scale_weights_poison * (model_vec - target_var) + target_var, target_var, dim=0)
        # cs_sim = cs_loss(model_vec, target_vec)

        loss = 1 - cs_sim

        return 1e3 * loss
    def get_one_vec(self, model):
        size = 0
        for name, layer in model.named_parameters():
            if name == 'decoder.weight':
                continue
            size += layer.view(-1).shape[0]
        sum_var = torch.tensor(torch.cuda.FloatTensor(size).fill_(0))

        size = 0
        for name, layer in model.named_parameters():
            if name == 'decoder.weight':
                continue

            sum_var[size:size + layer.view(-1).shape[0]] = (layer.data).view(-1)
            size += layer.view(-1).shape[0]

        return sum_var