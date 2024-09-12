from collections import OrderedDict

import numpy as np
import torch

from Aggregation.agg_fedavg import FedAvg


class AggTrimmedMean(FedAvg):
    def __init__(self, conf, TrimK=0.25):
        super(AggTrimmedMean, self).__init__(conf)
        self.conf = conf
        self.TrimK = TrimK

    def aggregate_weight_trimmedmean(self, model, clients, choice_id):
        TotalParams = {}
        for i in choice_id:
            client = clients[i]
            TotalParams[i] = client.local_model.state_dict()
        N = len(choice_id)
        K = int(N * self.TrimK)
        if K >= int(N / 2):
            K = int(N / 2) - 1

        averaged_weights = OrderedDict()
        for layer, weight in model.state_dict().items():
            averaged_weights[layer] = torch.zeros_like(weight).to(self.conf['DEVICE'])
        layers = averaged_weights.keys()
        C = 0
        for layer in layers:
            LayerParams = []
            for i in choice_id:
                LayerParams.append(TotalParams[i][layer].cpu().detach().numpy())
            SortLayerParams = np.sort(np.array(LayerParams), axis=0)
            GlobalTrimedParams = []
            for i in range(K, N - K):
                GlobalTrimedParams.append(SortLayerParams[i])
            GlobalParam = np.mean(GlobalTrimedParams, axis=0)
            GlobalParam = np.array(GlobalParam)
            averaged_weights[layer] = torch.from_numpy(GlobalParam).to(self.conf['DEVICE'])
        return averaged_weights, None
