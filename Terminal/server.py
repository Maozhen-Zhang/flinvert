import numpy as np


class Server():
    def __init__(self, cfg, model, tmp_model, defense):
        self.cfg = cfg
        self.global_model = model.to(cfg.device)
        self.tmp_model = tmp_model.to(cfg.device)
        self.defense = defense

    def aggregate_weight(self, weight_accumulators, choice_id):
        defense = self.defense
        model = self.global_model
        global_weight, AggIDs = defense.aggregation(model, weight_accumulators, choice_id)
        self.global_model.load_state_dict(global_weight)

    def getChoiceId(self):
        cfg = self.cfg
        sample = cfg.sample
        choice_id = sorted(np.random.choice([i for i in range(cfg.n_client)], sample, replace=False))
        mal_id = sorted([id for id in choice_id if id in cfg.mal_id])
        print(f"|---Train client id is {choice_id}")
        print(f"|---Mal client id is {mal_id}")

        return choice_id, mal_id
