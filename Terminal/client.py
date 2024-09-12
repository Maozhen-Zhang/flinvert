import copy

import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader

from Functions.data import get_dataset, get_data_indicates
from Functions.get_model import init_model
from Functions.log import configure_logger
from Models.init_model import CNNMnist
from Models.simple import SimpleNet


class Client:
    def __init__(self, cfg, ID, train_dataset, test_dataset, identity="None"):
        self.cfg = cfg
        self.ID = ID
        self.train_dataset = train_dataset
        self.dataloader = DataLoader(self.train_dataset, batch_size=cfg.batch_size, shuffle=True)
        self.test_dataset = test_dataset
        self.identity = identity
        self.local_model = None
        self.cfg.image_shape = self.train_dataset[0][0].shape



    def updateEpoch(self, e):
        self.current_epoch = e

    def load_model(self, model):
        self.local_model = model

    def uploadWeight(self, weight_accumulators):
        weight_accumulator = dict()
        for name, data in self.local_model.state_dict().items():
            weight_accumulator[name] = data
        weight_accumulators[self.ID] = weight_accumulator
        self.weight_accumulators = weight_accumulators




    def local_train(self, cfg):
        self.benign_train(cfg)

    def benign_train(self, cfg):
        lr = cfg.lr
        weight_decay = cfg.weight_decay
        momentum = cfg.momentum
        optimizer = optim.SGD(self.local_model.parameters(), lr=lr,
                                  weight_decay=weight_decay,
                                  momentum=momentum)
        dataloader = self.dataloader
        print(f"|---Client {self.ID} train (Benign Process)")
        self._train(cfg, self.local_model, dataloader, optimizer=optimizer)


    def _train(self, cfg, model, dataloader, optimizer=None):
        device = cfg.device
        epoch = cfg.local_epoch
        lr = cfg.lr
        weight_decay = cfg.weight_decay
        momentum = cfg.momentum
        if optimizer is None:
            optimizer = optim.SGD(model.parameters(), lr=lr,
                                  weight_decay=weight_decay,
                                  momentum=momentum)
        criterion = torch.nn.CrossEntropyLoss()
        model.train()
        model.to(device)
        for e in range(epoch):
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                if data.shape[0] == 1:
                    continue
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
