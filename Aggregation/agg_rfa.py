"""
    Pillutla, V.K., Kakade, S.M., Harchaoui, Z.: Robust aggregation for federated learning. CoRR (2019) arxiv:1912.13445
    we implement the robust aggregator at:
    https://arxiv.org/pdf/1912.13445.pdf
    the code is translated from the TensorFlow implementation:
    https://github.com/krishnap25/RFA/blob/01ec26e65f13f46caf1391082aa76efcdb69a7a8/models/model.py#L264-L298
"""
import copy
from collections import OrderedDict

import torch

from Aggregation.agg_fedavg import FedAvg
from Functions.log import get_logger

# TODO: something is wrong
class AggRFA(FedAvg):
    def __init__(self, conf, maxiter=4):
        super(AggRFA, self).__init__(conf)
        self.conf = conf
        self.maxiter = maxiter
        self.logger = get_logger(conf['logger']['logger_name'])

    def aggregateWeightRFA(self, PreGlobalModel, clients, chose_id):
        conf = self.conf
        DEVICE = conf['DEVICE']
        number_client_per_round = self.conf['sample_client']
        number_client_datasets = [len(clients[i].train_dataset) for i in chose_id]
        total_num_dps_per_round = sum(number_client_datasets)
        client_datasets_prop = [number_client_datasets[i] / total_num_dps_per_round for i in
                                range(number_client_per_round)]

        maxiter = self.maxiter
        eps = 1e-5
        ftol = 1e-7
        # Computes geometric median of atoms with weights alphas using Weiszfeld's Algorithm
        alphas = torch.tensor(client_datasets_prop, dtype=torch.float32, device=DEVICE)
        vectorize_nets = [self.vectorize_net(client.local_model).detach() for ID, client in clients.items()]
        median = self.weighted_average_oracle(vectorize_nets, alphas)

        num_oracle_calls = 1

        # logging
        obj_val = self.geometric_median_objective(median=median, points=vectorize_nets, alphas=alphas)

        logs = []
        log_entry = [0, obj_val, 0, 0]
        logs.append("Tracking log entry: {}".format(log_entry))
        self.logger.info('|---Starting Weiszfeld algorithm')
        self.logger.info(log_entry)

        # start
        for i in range(maxiter):
            prev_median, prev_obj_val = median, obj_val
            weights = torch.tensor(
                [alpha / max(eps, self.l2dist(median, p).item()) for alpha, p in zip(alphas, vectorize_nets)],
                dtype=alphas.dtype, device=DEVICE)
            weights = weights / weights.sum()
            median = self.weighted_average_oracle(vectorize_nets, weights)
            num_oracle_calls += 1
            obj_val = self.geometric_median_objective(median, vectorize_nets, alphas)
            log_entry = [i + 1, obj_val,
                         (prev_obj_val - obj_val) / obj_val,
                         self.l2dist(median, prev_median)]
            logs.append(log_entry)
            logs.append("Tracking log entry: {}".format(log_entry))
            self.logger.info("|--- Oracle Cals: {}, Objective Val: {}".format(num_oracle_calls, obj_val))
            if abs(prev_obj_val - obj_val) < ftol * obj_val:
                break
            # logger.info("Num Oracale Calls: {}, Logs: {}".format(num_oracle_calls, logs))

        # aggregated_model = client_models[0]  # create a clone of the model
        # self.load_model_weight(aggregated_model, median.to(DEVICE))
        # neo_net_list = [aggregated_model]
        # neo_net_freq = [1.0]

        NewGlobalModel = copy.deepcopy(PreGlobalModel)
        self.load_model_weight(NewGlobalModel, median.to(DEVICE))

        global_weight, AggIDs = self.aggregateByModel(PreGlobalModel, [NewGlobalModel], [0])

        return global_weight, AggIDs

    def load_model_weight(self, net, weight):
        index_bias = 0
        for p_index, p in enumerate(net.parameters()):
            p.data = weight[index_bias:index_bias + p.numel()].view(p.size())
            index_bias += p.numel()

    def vectorize_net(self, net):
        return torch.cat([p.view(-1) for p in net.parameters()])

    def weighted_average_oracle(self, points, weights):
        """Computes weighted average of atoms with specified weights
        Args:
            points: list, whose weighted average we wish to calculate
                Each element is a list_of_torch.Tensor
            weights: list of weights of the same length as atoms
        """
        DEVICE = self.conf['DEVICE']
        tot_weights = weights.sum()
        weighted_updates = torch.zeros(points[0].shape, dtype=points[0].dtype, device=points[0].device)
        weighted_updates = weighted_updates.to(DEVICE)
        for w, p in zip(weights, points):
            w = w.to(DEVICE)
            p = p.to(DEVICE)
            weighted_updates += (w * p / tot_weights)
        return weighted_updates

    def geometric_median_objective(self, median, points, alphas):
        """Compute geometric median objective."""
        return torch.sum(torch.stack([alpha * self.l2dist(median, p) for alpha, p in zip(alphas, points)]))

    def l2dist(self, p1, p2):
        """L2 distance between p1, p2, each of which is a list of nd-arrays"""
        DEVICE = self.conf['DEVICE']
        p2 = p1.to(DEVICE)
        p2 = p2.to(DEVICE)
        return torch.norm(p1 - p2)
