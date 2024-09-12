import copy
import math
import pdb

import numpy as np
import torch

from Aggregation.agg_fedavg import FedAvg
from Functions.log import get_logger



# ToDo: Add CRFL
class AggCRFL(FedAvg):
    def __init__(self, conf):
        super(AggCRFL, self).__init__(conf)
        self.conf = conf

        self.logger = get_logger(conf['logger']['logger_name'])

        self.TYPE_LOAN = 'loan'
        self.TYPE_MNIST = 'mnist'
        self.TYPE_EMNIST = 'emnist'
        self.TYPE_CIFAR10 = 'cifar10'
        self.TYPE_CIFAR100 = 'cifar100'
        self.TYPE_TINY_IMAGENET = 'tiny-imagenet'

    def aggregateWeightCRFL(self, PreGlobalModel, clients, choice_id):
        global_weight, AggIDs = self.aggregate_grad(PreGlobalModel, clients, choice_id)
        global_model = copy.deepcopy(PreGlobalModel)
        global_model.load_state_dict(global_weight)
        self.clip_weight_norm(global_model)
        self.add_differential_privacy_noise(global_model,sigma=0.002, cp=False)
        return global_model.state_dict(), -1


    def clip_weight_norm(self, model,clip=14):
        total_norm = self.get_global_model_norm(model)
        print("total_norm: " + str(total_norm) + "clip: " + str(clip))
        max_norm = clip
        clip_coef = max_norm / (total_norm + 1e-6)
        current_norm = total_norm
        if total_norm > max_norm:
            for name, layer in model.named_parameters():
                layer.data.mul_(clip_coef)
            current_norm = self.get_global_model_norm(model)
        return current_norm

    def get_global_model_norm(self, model):
        squared_sum = 0
        for name, layer in model.named_parameters():
            squared_sum += torch.sum(torch.pow(layer.data, 2))
        return math.sqrt(squared_sum)


    def add_differential_privacy_noise(self, model,sigma=0.001, cp=False):
        if not cp:
            for name, param in model.state_dict().items():
                if 'tracked' in name or 'running' in name:
                    continue
                # print(name)
                dp_noise = torch.cuda.FloatTensor(param.shape).normal_(mean=0, std=sigma)
                param.add_(dp_noise)
        else:
            smoothed_model = copy.deepcopy(model)
            for name, param in smoothed_model.state_dict().items():
                if 'tracked' in name or 'running' in name:
                    continue
                dp_noise = torch.cuda.FloatTensor(param.shape).normal_(mean=0, std=sigma)
                param.add_(dp_noise)
            return smoothed_model
