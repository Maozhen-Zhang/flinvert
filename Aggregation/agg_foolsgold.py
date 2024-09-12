import copy
import pdb
import time

import numpy as np
import torch

from Aggregation.agg_fedavg import FedAvg
from Functions.log import get_logger


class AggFoolsGold(FedAvg):
    def __init__(self, cfg):
        super(AggFoolsGold, self).__init__(cfg)
        self.cfg = cfg
        self.memory_size = 0  # The number of previous iterations to use FoolsGold on
        self.number_clients_per_round = self.cfg.sample
        self.n_classes = self.cfg.classes
        self.fg = FoolsGold(use_memory=True)


    def aggregateWeightFoolsGold(self, model, weight_accumulators, choice_id):
        client_grads = []
        for id in choice_id:
            grad = []
            for name, param in model.state_dict().items():
                grad.append(weight_accumulators[id][name] - param)
            client_grads.append(grad)
        wv = self.fg.aggregate_gradients(client_grads, choice_id, model.state_dict())
        return wv
        # weight_accumulator = dict()
        # print(f"|---foolsgold wv is: {wv}")
        # for name, data in model.state_dict().items():
        #     weight_accumulator[name] = torch.zeros_like(data)
        # for i, id in enumerate(choice_id):
        #     for name, data in weight_accumulators[id].items():
        #         grad_ = (data - model.state_dict()[name]) * wv[i] * (1 / len(choice_id))
        #         grad_ = torch.tensor(grad_, dtype=data.dtype)
        #         weight_accumulator[name].add_(grad_)
        #
        # lr = self.cfg.agglr
        # for name, data in model.state_dict().items():
        #     update_per_layer = weight_accumulator[name] * lr
        #     update_per_layer = torch.tensor(update_per_layer, dtype=data.dtype)
        #     data.add_(update_per_layer.to(self.cfg.device))

class FoolsGold(object):
    def __init__(self, use_memory=False):
        self.memory = None
        self.memory_dict = dict()
        self.wv_history = []
        self.use_memory = use_memory

    def aggregate_gradients(self, client_grads, choice_id, global_weight):
        cur_time = time.time()
        num_clients = len(choice_id)
        grad_len = np.array(client_grads[0][-2].cpu().data.numpy().shape).prod()

        # if self.memory is None:
        #     self.memory = np.zeros((num_clients, grad_len))
        self.memory = np.zeros((num_clients, grad_len))
        grads = np.zeros((num_clients, grad_len))

        for i in range(len(client_grads)):
            grads[i] = np.reshape(client_grads[i][-2].cpu().data.numpy(), (grad_len))
            if choice_id[i] in self.memory_dict.keys():
                self.memory_dict[choice_id[i]] += grads[i]
            else:
                self.memory_dict[choice_id[i]] = copy.deepcopy(grads[i])
            self.memory[i] = self.memory_dict[choice_id[i]]

        if self.use_memory:
            wv, alpha = self.foolsgold(self.memory)  # Use FG
        else:
            wv, alpha = self.foolsgold(grads)  # Use FG
        # logger.info(f'[foolsgold agg] wv: {wv}')
        self.wv_history.append(wv)
        print(f'[foolsgold agg] wv: {wv}')
        return wv

    def foolsgold(self, grads):
        """
        :param grads:
        :return: compute similatiry and return weightings
        """
        n_clients = grads.shape[0]
        import sklearn.metrics.pairwise as smp
        cs = smp.cosine_similarity(grads) - np.eye(n_clients)

        maxcs = np.max(cs, axis=1)
        # pardoning
        for i in range(n_clients):
            for j in range(n_clients):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
        wv = 1 - (np.max(cs, axis=1))

        wv[wv > 1] = 1
        wv[wv < 0] = 0

        alpha = np.max(cs, axis=1)

        # Rescale so that max value is wv
        wv = wv / np.max(wv)
        wv[(wv == 1)] = .99

        # Logit function
        wv = (np.log(wv / (1 - wv)) + 0.5)
        wv[(np.isinf(wv) + wv > 1)] = 1
        wv[(wv < 0)] = 0

        # wv is the weight
        return wv, alpha
