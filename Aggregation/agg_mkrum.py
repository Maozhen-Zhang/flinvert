import copy
from collections import defaultdict

import numpy as np
import torch

from Aggregation.agg_fedavg import FedAvg
from Functions.log import get_logger


class AggMKrum(FedAvg):
    def __init__(self, cfg):
        super(AggMKrum, self).__init__(cfg)
        self.cfg = cfg

    def getBenignParams(self, model, weight_accumulators, choice_id):
        TotalParams = {i: weight_accumulators[i] for i in choice_id}
        GlobalParas, FindIDs = self.agg_krum(TotalParams, Frac=4)
        return GlobalParas, FindIDs

    def agg_krum(self, Paras, Frac=4, Num=1):
        """
        :param Param:
        :param Frac: the frac of the number of the mal clients
        :param num:
        :return:
        """
        if self.cfg.defense == 'krum':
            Num = 1
        else:
            Num = 6
        N = len(Paras)
        f = Frac
        M = N - f
        if M <= 1:
            M = N
        Distances = defaultdict(dict)
        client_ids = Paras.keys()
        client_ids = list(client_ids)
        Kys = Paras[client_ids[0]].keys()
        for i, ClientID1 in enumerate(client_ids):
            Param1 = Paras[ClientID1]
            for j, ClientID2 in enumerate(client_ids[i:]):
                Param2 = Paras[ClientID2]
                distance = 0
                if ClientID1 != ClientID2:
                    for ky in Kys:
                        if 'weight' in ky or 'bias' in ky:
                            distance += np.linalg.norm(
                                Param1[ky].cpu().detach().numpy() - Param2[ky].cpu().detach().numpy()) ** 2
                    distance = np.sqrt(distance)
                Distances[ClientID1][ClientID2] = distance
                Distances[ClientID2][ClientID1] = distance

        if Num == 1:
            FindID = -1
            FindVal = pow(10, 20)
            PDict = {}
            for ClientID in client_ids:
                Dis = sorted(Distances[ClientID].values())
                SumDis = np.sum(Dis[:M])
                PDict[ClientID] = SumDis
                if FindVal > SumDis:
                    FindVal = SumDis
                    FindID = ClientID
            return Paras[FindID], [FindID]

        if Num >= 2:
            Dist = copy.deepcopy(Distances)
            PDict = {}
            for ClientID in client_ids:
                Dis = sorted(Dist[ClientID].values())
                SumDis = np.sum(Dis[:M])
                PDict[ClientID] = SumDis
            SDict = sorted(PDict.items(), key=lambda x: x[1])

            GParas = {}
            FindIDs = []
            for i in range(Num):
                Ky = SDict[i][0]
                GParas[i] = Paras[Ky]
                FindIDs.append(Ky)
            return GParas, FindIDs
