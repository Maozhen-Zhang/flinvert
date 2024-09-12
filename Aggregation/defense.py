
from Aggregation.agg_crfl import AggCRFL
from Aggregation.agg_deepsight import AggDeepSight
from Aggregation.agg_dp import AggDP
from Aggregation.agg_fedavg import FedAvg
from Aggregation.agg_foolsgold import AggFoolsGold
from Aggregation.agg_median import AggMedian
from Aggregation.agg_mkrum import AggMKrum

from Aggregation.agg_rlr import AggRLR
from Aggregation.agg_trimmedmean import AggTrimmedMean


class Defense:
    def __init__(self, cfg):
        self.cfg = cfg

        self.defense_map = {
            'fedavg': FedAvg,
            'krum': lambda conf: AggMKrum(conf, 1),
            'mkrum': lambda conf: AggMKrum(conf, 6),
            'Trimmedmean': AggTrimmedMean,
            'Median': AggMedian,
            'RLR': AggRLR,
            'DeepSight': AggDeepSight,
            'FoolsGold': AggFoolsGold,
            'CRFL': AggCRFL,

            # 'CDP': lambda conf: AggDP(conf),
            # 'LDP': lambda conf: AggDP(conf),
            # 'WeakDP': lambda conf: AggDP(conf),
            # 'NormClip': lambda conf: AggDP(conf),
        }

        self.method = self.get_defense_method(cfg)

    def setEpochInfo(self, e):
        self.epoch = e
        self.method.setEpochInfo(e)

    def get_defense_method(self, cfg):
        defense = cfg.defense
        defenses = {
            "fedavg": FedAvg,
            # "krum" :  AggMKrum,
            "mkrum": AggMKrum,
            "deepsight": AggDeepSight,
            "foolsgold": AggFoolsGold,
            "rlr": AggRLR,
        }
        return defenses[defense](cfg)


    def aggregation(self, model, weight_accumulators, choice_id):
        defense = self
        method = self.cfg.defense
        if  method == 'FedAvg' or method == "fedavg":
            global_weight, AggIDs = defense.method.aggregate_grad(model, weight_accumulators, choice_id)
        elif method == 'mkrum' or method == 'krum':
            BenignParams, AggIDs = defense.method.getBenignParams(model, weight_accumulators, choice_id)
            global_weight, AggIDs = defense.method.aggregate_grad(model, weight_accumulators, AggIDs)
        # elif method == 'Trimmedmean':
        #     global_weight, AggIDs = defense.method.aggregate_weight_trimmedmean(model, weight_accumulators, choice_id)
        #     global_weight, AggIDs = defense.method.aggregate_grad_diff(global_weight, model.state_dict())
        # elif method == 'Median':
        #     global_weight, AggIDs = defense.method.aggregateWeightMedian(model, weight_accumulators, choice_id)
        #     global_weight, AggIDs = defense.method.aggregate_grad_diff(global_weight, model.state_dict())
        elif method == 'RLR' or method == 'rlr':
            global_weight, AggIDs = defense.method.aggregateWeightRLR(model, weight_accumulators, choice_id, robustLR_threshold=2)
        elif method == 'DeepSight' or method == 'deepsight':
            global_weight, AggIDs = defense.method.aggregateDeepSight(model, weight_accumulators, choice_id)
        # elif method == 'CDP':
        #     global_weight, AggIDs = defense.method.aggregateCDP(model, weight_accumulators, choice_id)
        # elif method == 'LDP':
        #     global_weight, AggIDs = defense.method.aggregateLDP(model, weight_accumulators, choice_id)
        # elif method == 'WeakDP':
        #     global_weight, AggIDs = defense.method.aggregateWeakDP(model, weight_accumulators, choice_id)
        # elif method == 'NormClip':
        #     global_weight, AggIDs = defense.method.aggregateNormClip(model, weight_accumulators, choice_id)
        elif method == 'CRFL':  # TODO
            global_weight, AggIDs = defense.method.aggregateWeightCRFL(model, weight_accumulators, choice_id)
        elif method == 'FoolsGold' or method == 'foolsgold':
            wv = defense.method.aggregateWeightFoolsGold(model, weight_accumulators, choice_id)
            global_weight, AggIDs = defense.method.aggregate_grad(model, weight_accumulators, choice_id, pts = wv)
        # elif method == 'RFA':
        #     global_weight, AggIDs = defense.method.aggregateWeightRFA(model, weight_accumulators, choice_id)
        else:
            assert (2 == 1)

        return global_weight, AggIDs
