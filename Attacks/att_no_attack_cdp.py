import copy
import statistics

import torch

from Attacks.att_no_attack import NoAtt


class NoAttCDP(NoAtt):
    def __init__(self, conf):
        super(NoAttCDP, self).__init__(conf)
        self.conf = conf
    def compute_local_model_update_(self, PreModel, LastModel):
        if len(list(PreModel.parameters())) != len(list(LastModel.parameters())):
            raise AssertionError("Two models have different length in number of clients")
        norm_values = []
        for i in range(len(list(PreModel.parameters()))):
            norm_values.append((list(LastModel.parameters())[i] - list(PreModel.parameters())[i]))
        norm_flatten = torch.cat([torch.flatten(i) for i in norm_values])
        return torch.norm(norm_flatten).item()
        # norm_values = []
        # for i in range(len(list(PreModel.parameters()))):
        #     norm_values.append(torch.norm(list(LastModel.parameters())[i] - list(PreModel.parameters())[i]).item())
        # return sum(norm_values)
    def compute_local_model_update(self, PreModel, LastModel):
        if len(list(PreModel.parameters())) != len(list(LastModel.parameters())):
            raise AssertionError("Two models have different length in number of clients")
        norm_values = []
        for i in range(len(list(PreModel.parameters()))):
            norm_values.append(torch.norm(list(PreModel.parameters())[i] - list(LastModel.parameters())[i]))
        return statistics.median(norm_values)

    def train(self, epoch, model, trainloader, optimizer, criterion, DEVICE):
        DEVICE = self.conf['DEVICE']
        first_model = copy.deepcopy(model)
        model.train()
        epoch_loss = []
        model.to(DEVICE)
        for e in range(epoch):
            for batch_idx, (data, target) in enumerate(trainloader):
                data, target = data.to(DEVICE), target.to(DEVICE)
                if data.shape[0] == 1:
                    continue
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        # l2norm = self.compute_local_model_update(first_model, model)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), l2norm)
        # self.l2norm = l2norm
        # optimizer.step()
