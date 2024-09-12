import copy

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from Aggregation.agg_fedavg import FedAvg
from Functions.log import get_logger

"""
    Byzantine-Robust Federated Machine Learning through Adaptive Model Averaging
"""


class AggFineTuning(FedAvg):
    def __init__(self, conf,dataset):
        super(AggFineTuning,self).__init__()
        self.conf = conf
        self.trainloader = DataLoader(dataset, batch_size=self.conf['batch_size'], shuffle=True, num_workers=0, drop_last=True)
        self.logger = get_logger(conf['logger']['logger_name'])

    def aggregateWeightFineTuning(self,model,clients,chosen_ids, pts=None):
        NewGloablWeight, chosen_ids = self.aggregate_grad(model, clients, chosen_ids, pts)
        trainloader = self.trainloader
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.conf['lr'],
                              weight_decay=self.conf['weight_decay'],
                              momentum=self.conf['momentum'])
        DEVICE = self.conf['DEVICE']
        model.train()
        epoch_loss = []
        model.to(DEVICE)
        for e in range(1):
            batch_loss = []
            for batch_idx, (data, target) in enumerate(trainloader):
                data, target = data.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
