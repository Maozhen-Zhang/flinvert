import statistics

import torch

from Attacks.attrobust import AttRobust


class NoAtt():
    def __init__(self, conf):
        self.conf = conf
        self.mask = None
        self.pattern = None

    def setEpochInfo(self, e):
        self.current_epoch = e

    def setL2Norm(self, l2norm):
        self.l2norm = l2norm

    def getL2Norm(self):
        return self.l2norm

    def train(self, epoch, model, trainloader, optimizer, criterion, DEVICE):
        DEVICE = self.conf['DEVICE']
        model.train()
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