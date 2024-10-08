import math
from collections import OrderedDict

import numpy as np
import torch
from torch.autograd import Variable

from Functions.log import get_logger


class FedAvg:
    def __init__(self, cfg):
        self.cfg = cfg

    def setEpochInfo(self, e):
        self.current_epoch = e


    def aggregate_grad(self, model, weight_accumulators:OrderedDict, chosen_ids, pts=None):
        assert not weight_accumulators is None and len(weight_accumulators.keys()) > 0

        device = self.cfg.device

        PreGlobalWeight = model.state_dict()
        UpdateGlobalWeight = model.state_dict()

        if pts is None:
            pts = {i: 1.0 / len(chosen_ids) for i in chosen_ids}
        else:
            pts = {id: pts[i]/ len(chosen_ids)  for i, id in enumerate(chosen_ids)}

        # init averaged_grad
        averaged_grad = OrderedDict()

        for layer, weight in PreGlobalWeight.items():
            averaged_grad[layer] = torch.zeros_like(weight).to(device)

        # compute gradient diff
        for i in chosen_ids:
            local_model_weight = weight_accumulators[i]
            diffgrads = {}
            for layer, weight in local_model_weight.items():
                prop = torch.tensor(pts[i]).to(weight.dtype).to(device)
                diffgrads[layer] = (weight - PreGlobalWeight[layer]).detach()
                averaged_grad[layer] += diffgrads[layer] * prop

        # gradient updates
        alpha_lr = self.cfg.agglr
        for layer, weight in averaged_grad.items():
            alpha = torch.tensor(alpha_lr).to(weight.dtype).to(device)
            averaged_grad[layer] = weight * alpha

        # updates Global Weight
        for layer, weight in averaged_grad.items():
            UpdateGlobalWeight[layer] += averaged_grad[layer]
        return UpdateGlobalWeight, chosen_ids