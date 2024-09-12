import copy

import numpy as np
import torch
import yaml

from Attacks.attbackdoor import AttBackdoor
from Functions.log import get_logger, configure_logger

"""
    How To Backdoor Federated Learning
"""


class AttNeurotoxin(AttBackdoor):
    def __init__(self, conf, MalID=None):
        super().__init__(conf)
        self.conf = conf
        self.logger = get_logger(conf['logger']['logger_name'])
        self.setMalSettingPerRount(MalID)
        self.initTrigger()

    def setMalSettingPerRount(self, MalID=None):
        self.MalNumPerRound = self.conf['MalSetting']['MalNumPerRound']
        self.MalIDs = self.conf['MalSetting']['MalIDs']
        self.MalID = MalID

    def initTrigger(self):
        DEVICE = self.conf['DEVICE']
        self.initPattern()
        self.initMask()

        self.mask = self.mask.to(DEVICE)
        self.Totalmask = self.Totalmask.to(DEVICE)

        if self.MalID == -1:
            self.logger.debug(f"|---Total Mask init is---:\n" + str(self.Totalmask[3:6, 23:26]))
            # self.logger.debug(f"|---Total trigger init is---:\n" + str((self.mask * self.pattern)[:, 3:7, 3:13]))

        if self.MalID in self.MalIDs:
            self.logger.debug(f"|---MalID  is---{self.MalID}")
            self.logger.debug(f"|---local Mask init is---:\n" + str(self.mask[3:6, 23:26]))
            self.logger.debug(f"|---Local trigger init is---:\n" + str((self.mask * self.pattern)[:, 3:6, 23:26]))

    def initMask(self):
        DEVICE = self.conf['DEVICE']

        self.TotalPoisonLocation = [
            [3, 23], [3, 24], [3, 25],
            [4, 23], [4, 24], [4, 25],
            [5, 23], [5, 24], [5, 25],
        ]
        self.PoisonLocation = self.TotalPoisonLocation
        self.Totalmask = torch.zeros(self.conf['ImageShape'][-2:]).to(DEVICE)
        for xloc, yloc in self.TotalPoisonLocation:
            self.Totalmask[xloc, yloc] = 1

        self.mask = torch.zeros(self.conf['ImageShape'][-2:]).to(DEVICE)
        if self.MalID != None and self.MalID in self.MalIDs:
            PoisonLocation = self.PoisonLocation
            for poison_x, poison_y in PoisonLocation:
                self.mask[poison_x, poison_y] = 1
        elif self.MalID == -1:
            self.mask = self.Totalmask

    def initPattern(self):
        DEVICE = self.conf['DEVICE']
        dataset = self.conf['dataset']

        if dataset == 'mnist' or dataset == 'cifar10':
            self.pattern = torch.ones(self.conf['ImageShape']).to(DEVICE)
            PatternValues = self.UpperBound
            for channel in range(len(PatternValues)):
                self.pattern[channel].fill_(PatternValues[channel]).to(DEVICE)
        # self.pattern = self.getNormalize(self.pattern)

    def get_grad_mask_on_cv(self, local_model, train_loader, ratio=0.9):
        """Generate a gradient mask based on the given dataset, in the experiment we apply ratio=0.9 by default"""
        DEVICE = self.conf['DEVICE']
        model = local_model
        model.train()

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

    def train(self, epoch, model, trainloader, optimizer, criterion, DEVICE):
        conf = self.conf
        ScaleLambda = conf[conf['attack']]['ScaleLambda']
        SimLambda = conf[conf['attack']]['SimLambda']
        raw_model = copy.deepcopy(model)
        mask_grad_list = self.get_grad_mask_on_cv(model, trainloader)
        for e in range(epoch):
            for batch_idx, (data, target) in enumerate(trainloader):
                if data.shape[0] == 1:
                    data = torch.concat([data, data])
                    target = torch.concat([target, target])
                data, target = data.to(DEVICE), target.to(DEVICE)
                poison_data, poison_target = self.injectTrigger2Imgs(data, target)
                # print(poison_data[0,0,3:9,23:26])
                # print(poison_data[0,0,3:6,3:10])

                optimizer.zero_grad()

                # model.zero_grad()
                output = model(data)
                loss1 = criterion(output, target).mean()

                # model.zero_grad()
                poison_output = model(poison_data)

                loss2 = criterion(poison_output, poison_target).mean()
                loss = ScaleLambda[0] * loss1 + ScaleLambda[1] * loss2
                loss += SimLambda * self.model_similarity_loss(raw_model, model)
                loss.backward()
                self.apply_grad_mask(model, mask_grad_list)
                optimizer.step()

        # diff_norm = Helper.model_dist_norm(raw_mode, model.state_dict())
        # self.logger.info(f"|---更新后距离, {diff_norm}")


if __name__ == '__main__':
    with open(f'../configs/conf.yaml', "r+") as file:
        conf = yaml.load(file, Loader=yaml.FullLoader)
    configure_logger(conf)
    conf['DEVICE'] = 'mps'
    conf['ImageShape'] = (3, 32, 32)
    att = AttCBA(conf, -1)
    att.initTrigger()
