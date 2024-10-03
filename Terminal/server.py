import numpy as np

from Aggregation.function_normclip import Normclip


class Server():
    def __init__(self, cfg, model, tmp_model, defense):
        self.cfg = cfg
        self.global_model = model.to(cfg.device)
        self.tmp_model = tmp_model.to(cfg.device)
        self.defense = defense
        self.normclip = Normclip()
    def aggregate_weight(self, weight_accumulators, choice_id):
        defense = self.defense
        model = self.global_model
        global_weight, AggIDs = defense.aggregation(model, weight_accumulators, choice_id)
        self.global_model.load_state_dict(global_weight)

    def getChoiceId(self, clients):
        cfg = self.cfg
        sample = cfg.sample
        choice_id = sorted(np.random.choice(list(clients.keys()), sample, replace=False))
        mal_id = sorted([id for id in choice_id if id in cfg.mal_id])
        print(f"|---Train client id is {choice_id}")
        print(f"|---Mal client id is {mal_id}")

        return choice_id, mal_id
