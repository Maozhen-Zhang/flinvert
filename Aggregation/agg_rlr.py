from collections import OrderedDict

import torch

from Aggregation.agg_fedavg import FedAvg
from Functions.log import get_logger


class AggRLR(FedAvg):
    def __init__(self, cfg):
        super(AggRLR, self).__init__(cfg)
        self.cfg = cfg

    def aggregateWeightRLR(self, model, weight_accumulators, chosen_ids, pts=None, robustLR_threshold=2):
        original_params = model.state_dict()


        # collect client updates
        updates = list()
        for id in chosen_ids:
            local_params = weight_accumulators[id]
            update = OrderedDict()
            for layer, weight in local_params.items():
                update[layer] = local_params[layer] - original_params[layer]
            updates.append(update)

        # compute_total_update
        robust_lrs = self.compute_robustLR(model, updates, robustLR_threshold=robustLR_threshold)
        # count signs：
        flip_analysis = dict()
        for layer in robust_lrs.keys():
            n_flip = torch.sum(torch.gt(robust_lrs[layer], 0.0).int())  # 大于零的符号
            n_unflip = torch.sum(torch.lt(robust_lrs[layer], 0.0).int())  # 小于0的值
            flip_analysis[layer] = [n_flip, n_unflip]

        for i, id in enumerate(chosen_ids):
            prop = 1/len(chosen_ids) * self.cfg.agglr
            self.robust_lr_add_weights(original_params, robust_lrs, updates[i], prop)
        return original_params, -1

    def compute_robustLR(self, model, updates, robustLR_threshold=2):
        layers = updates[0].keys()
        # signed_weights = OrderedDict()
        robust_lrs = OrderedDict()
        for layer, weight in model.state_dict().items():
            # signed_weights[layer] = torch.zeros_like(weight)
            robust_lrs[layer] = torch.zeros_like(weight)

        for layer in layers:  # 按层进行的 robust learning rate
            for update in updates:
                robust_lrs[layer] += torch.sign(update[layer])
            # import pdb
            # pdb.set_trace()
            robust_lrs[layer] = torch.abs(robust_lrs[layer])
            robust_lrs[layer][robust_lrs[layer] >= robustLR_threshold] = 1.0
            robust_lrs[layer][robust_lrs[layer] != 1.0] = -1.0
        return robust_lrs

    def robust_lr_add_weights(self, original_params, robust_lrs, update, prop):
        for layer in original_params.keys():
            if 'running' in layer or 'tracked' in layer:
                original_params[layer] = original_params[layer] + update[layer] * prop
            else:
                original_params[layer] = original_params[layer] + update[layer] * prop * robust_lrs[layer]
