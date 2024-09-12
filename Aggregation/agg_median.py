from collections import OrderedDict
from typing import List

import torch

from Aggregation.agg_fedavg import FedAvg
from Functions.log import get_logger


class AggMedian(FedAvg):
    def __init__(self, conf):
        self.conf = conf
        self.logger = get_logger(conf['logger']['logger_name'])

    def _mean(self, inputs: List[torch.Tensor]):
        inputs_tensor = torch.stack(inputs, dim=0)
        return inputs_tensor.mean(dim=0)

    def _median(self, inputs: List[torch.Tensor]):
        inputs_tensor = torch.stack(inputs, dim=0)
        values_upper, _ = inputs_tensor.median(dim=0)
        values_lower, _ = (-inputs_tensor).median(dim=0)
        return (values_upper - values_lower) / 2

    def aggregateWeightMedian(self, model, clients, choice_id):
        DEVICE = self.conf['DEVICE']
        TotalParams = {}
        TotalParamsList = []
        for i in choice_id:
            client = clients[i]
            Tmp = []
            TotalParams[i] = client.local_model.state_dict()
            for key, value in TotalParams[i].items():
                Tmp.append(value)
            TotalParamsList.append(Tmp)

        LayersParams = {}
        for layer, weight in model.state_dict().items():
            LayerParam = []
            for clientID in choice_id:
                client = clients[clientID]
                ClientParams = client.local_model.state_dict()
                LayerParam.append(ClientParams[layer])
            LayersParams[layer] = LayerParam
        AverageWeight = OrderedDict()
        for layer, weight in model.state_dict().items():
            AverageWeight[layer] = torch.zeros_like(weight).to(DEVICE)
        for layer, param in AverageWeight.items():
            AverageWeight[layer] = self._median(LayersParams[layer])
        return AverageWeight, -1
