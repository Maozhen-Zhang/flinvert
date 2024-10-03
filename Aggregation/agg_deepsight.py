import copy

import numpy as np
import torch
from sklearn.cluster import DBSCAN

from Aggregation.agg_fedavg import FedAvg
from Functions.log import get_logger


class AggDeepSight(FedAvg):
    def __init__(self, cfg):
        super(AggDeepSight, self).__init__(cfg)
        self.cfg = cfg

    def aggregateDeepSight(self, global_model, weight_accumulators, chosen_ids):
        cfg = self.cfg
        def ensemble_cluster(neups, ddifs, biases):
            biases = np.array([bias.cpu().numpy() for bias in biases])
            # neups = np.array([neup.cpu().numpy() for neup in neups])
            # ddifs = np.array([ddif.cpu().detach().numpy() for ddif in ddifs])
            N = len(neups)
            # use bias to conduct DBSCAM
            # biases= np.array(biases)

            # dbscan = DBSCAN(min_samples=3, metric='cosine')
            # cluster_labels = dbscan.fit_predict(biases)

            cosine_labels = DBSCAN(min_samples=3, metric='cosine').fit(biases).labels_
            print("cosine_cluster:{}".format(cosine_labels))
            # neups=np.array(neups)
            neup_labels = DBSCAN(min_samples=3).fit(neups).labels_
            print("neup_cluster:{}".format(neup_labels))
            ddif_labels = DBSCAN(min_samples=3).fit(ddifs).labels_
            print("ddif_cluster:{}".format(ddif_labels))

            dists_from_cluster = np.zeros((N, N))
            for i in range(N):
                for j in range(i, N):
                    dists_from_cluster[i, j] = (int(cosine_labels[i] == cosine_labels[j]) + int(
                        neup_labels[i] == neup_labels[j]) + int(ddif_labels[i] == ddif_labels[j])) / 3.0
                    dists_from_cluster[j, i] = dists_from_cluster[i, j]

            print("dists_from_clusters:")
            print(dists_from_cluster)
            ensembled_labels = DBSCAN(min_samples=3, metric='precomputed').fit(dists_from_cluster).labels_

            return ensembled_labels

        # 最后一层的权重和偏置
        global_weight = list(global_model.state_dict().values())[-2]
        global_bias = list(global_model.state_dict().values())[-1]
        # 每个客户端最后一层的权重和偏置 - 全局模型的权重和偏置
        weights = [list(weight_accumulators[i].values())[-2] for i in chosen_ids]
        biases = [(list(weight_accumulators[i].values())[-1] - global_bias) for i in chosen_ids]


        # n_client = len(chosen_ids)
        # cosine_similarity_dists = np.array((n_client, n_client))

        neups = list()
        n_exceeds = list()

        # calculate neups and n_exceeds
        sC_nn2 = 0
        for i in range(len(chosen_ids)):
            C_nn = torch.sum(weights[i] - global_weight, dim=[1]) + biases[i] - global_bias
            # print("C_nn:",C_nn)
            C_nn2 = C_nn * C_nn
            neups.append(C_nn2)
            sC_nn2 += C_nn2

            C_max = torch.max(C_nn2).item()
            threshold = 0.01 * C_max if 0.01 > (1 / len(biases)) else 1 / len(biases) * C_max
            n_exceed = torch.sum(C_nn2 > threshold).item()
            n_exceeds.append(n_exceed)

        # normalize
        neups = np.array([(neup/sC_nn2).cpu().numpy() for neup in neups])
        print("n_exceeds:{}".format(n_exceeds))

        rand_input = torch.randn((128, 3, cfg.img_size, cfg.img_size)).to(self.cfg.device)
        global_ddif = torch.mean(torch.softmax(global_model(rand_input), dim=1), dim=0)
        tmp_model = copy.deepcopy(global_model)

        # client_ddifs = [torch.mean(torch.softmax(tmp_model.load_state_dict(weight_accumulators[i])(rand_input), dim=1), dim=0)/ global_ddif
        #                 for i in chosen_ids]
        client_ddifs = []
        for i in chosen_ids:
            tmp_model.eval()
            # copy_params(tmp_model, weight_accumulators[i])
            tmp_model.load_state_dict(weight_accumulators[i])
            client_ddifs.append(torch.mean(torch.softmax(tmp_model(rand_input), dim=1), dim=0) / global_ddif)

        client_ddifs = np.array([client_ddif.cpu().detach().numpy() for client_ddif in client_ddifs])


        # use n_exceed to label
        # 计算一个下界
        classification_boundary = np.median(np.array(n_exceeds)) / 2
        # 值小于下届的是恶意客户端，为1的是恶意客户端
        identified_mals = [int(n_exceed <= classification_boundary) for n_exceed in n_exceeds]
        print("identified_mals:{}".format(identified_mals))
        clusters = ensemble_cluster(neups, client_ddifs, biases)
        print("ensemble clusters:{}".format(clusters))
        cluster_ids = np.unique(clusters)

        deleted_cluster_ids = list()

        # 找到恶意簇，统计被划分为恶意簇的客户端数量，如果超过1/3，则删除该簇
        for cluster_id in cluster_ids:
            n_mal = 0
            cluster_size = np.sum(cluster_id == clusters)   # 每个聚类的集群数量
            for identified_mal, cluster in zip(identified_mals, clusters):
                if cluster == cluster_id and identified_mal:
                    n_mal += 1
            print("cluser size:{} n_mal:{}".format(cluster_size,n_mal))
            if (n_mal / cluster_size) >= (1 / 3):
                deleted_cluster_ids.append(cluster_id)
        # print("deleted_clusters:",deleted_cluster_ids)
        temp_chosen_ids = copy.deepcopy(chosen_ids)
        for i in range(len(chosen_ids)-1, -1, -1):
            # print("cluster tag:",clusters[i])
            if clusters[i] in deleted_cluster_ids:
                del chosen_ids[i]

        print("final clients length:{}".format(len(chosen_ids)))
        if len(chosen_ids)==0:
            chosen_ids = temp_chosen_ids
        return self.aggregate_grad(global_model, weight_accumulators, chosen_ids)