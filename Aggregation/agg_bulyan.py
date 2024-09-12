from Aggregation.agg_median import AggMedian
from Aggregation.agg_mkrum import AggMKrum
from Functions.log import get_logger


class AggBulyan:
    def __init__(self, conf):
        self.conf = conf
        self.logger = get_logger(conf['logger']['logger_name'])
        self.MKrum = AggMKrum(conf, num = 6)
        self.Median = AggMedian(conf)


    def aggregateBulyan(self, global_model, clients, chosen_ids):
        global_model, AggIDs = self.MKrum.getBenignParams(global_model, clients, chosen_ids)
        global_model, _ = self.Median.aggregate_grad(global_model, clients, AggIDs)
        return global_model, -1