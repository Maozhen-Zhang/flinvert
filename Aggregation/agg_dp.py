import torch

from Aggregation.agg_fedavg import FedAvg
from DP.differential_privacy.privacy_accountant.pytorch import accountant

from Functions.log import get_logger
import statistics
from torch.autograd import Variable

"""
    You Are Catching My Attention: Are Vision Transformers Bad Learners Under Backdoor Attacks?
    Local and Central Differential Privacy for Robustness and Privacy in Federated Learning
"""


class AggDP(FedAvg):
    def __init__(self, conf, identity=None):
        super(AggDP, self).__init__(conf)
        self.conf = conf
        self.logger = get_logger(conf['logger']['logger_name'])
        self.identity = identity

        if self.conf['defense'] == 'WeakDP':
            """
                作者测试了
                    \epsilon=3,\delta=10^{-5}
                    \epsilon=2.8,\delta=10^{-5}
            """
            self.noise_scale = 3
            self.sigma = 0.8
            self.delta = 10 ** (-5)
            self.max_epsilon = 8
            self.epsilon = 10
            self.frac = conf['sample_client'] / conf['n_client']
            self.num_users = conf['n_client']
            self.priv_accountant = accountant.GaussianMomentsAccountant(self.num_users)


        if self.conf['defense'] == 'CDP':
            # 1.4、3、5、10、20
            # 3 spent_eps=14.69
            # 5 spent_eps=7.981562432345018
            self.noise_scale = 5
            self.delta = 10 ** (-5)
            self.frac = conf['sample_client'] / conf['n_client']
            self.num_users = conf['n_client']
            self.priv_accountant = accountant.GaussianMomentsAccountant(self.num_users)

    def aggregateNormClip(self, model, clients, choice_id):
        global_weight, AggIDs = self.aggregate_grad(model, clients, choice_id)
        return global_weight, AggIDs

    def aggregateWeakDP(self, model, clients, choice_id):
        priv_accountant = self.priv_accountant
        l2norms = {}
        for ID in choice_id:
            client = clients[ID]
            l2norms[ID] = client.attack.method.getL2Norm()

        sensitivity = self.conf[self.conf['defense']]['sensitivity']
        sigma = self.conf[self.conf['defense']]['sigma']
        tmp_global_weight, AggIDs = self.aggregate_grad(model, clients, choice_id)
        model.load_state_dict(tmp_global_weight)
        self.add_noise(model, sigma, sensitivity)
        print("|---sensitivity is ", sensitivity)
        print("|---sigma is ", sigma)
        global_weight = model.state_dict()
        return global_weight, AggIDs

    def aggregateLDP(self,model, clients, choice_id):
        global_weight, AggIDs = self.aggregate_grad(model, clients, choice_id)
        return global_weight, AggIDs



    def aggregateCDP(self, model, clients, choice_id):
        conf = self.conf
        priv_accountant = self.priv_accountant
        l2norms = {}
        for ID in choice_id:
            client = clients[ID]
            l2norms[ID] = client.attack.method.getL2Norm()

        sensitivity = statistics.median(list(l2norms.values()))
        sigma = (sensitivity * self.noise_scale) / (self.frac * self.num_users)
        self.logger.info(f"|---sensitivity is {sensitivity}")
        self.logger.info(f"|---sigma is {sigma}")
        tmp_global_weight, AggIDs = self.aggregate_grad(model, clients, choice_id)
        model.load_state_dict(tmp_global_weight)
        self.add_noise(model, sigma, sensitivity)
        global_weight = model.state_dict()

        m = max(int(self.frac * self.num_users), 1)
        priv_accountant.accumulate_privacy_spending(self.noise_scale, m)
        priv = priv_accountant.get_privacy_spent(target_deltas=[self.delta])

        self.logger.info(f"|---Privacy Cost is {priv}")
        # if priv_accountant.get_privacy_spent(target_deltas=[args.delta])[0].spent_delta > args.max_epsilon:
        #     break
        return global_weight, AggIDs

    def add_noise(self, model, sigma, sensitivity):
        with torch.no_grad():
            for param in model.parameters():
                noise = self.compute_gaussian_noise(param.data, sigma, sensitivity)
                # print(noise.max())
                param += noise

    def compute_gaussian_noise(self, data, sigma, sensitivity):
        shape = data.shape
        # noise = Variable(torch.zeros(shape))
        noise = torch.zeros(shape)
        noise.data.normal_(0.0, std=sigma * sigma)
        return noise.to('cuda:0')
