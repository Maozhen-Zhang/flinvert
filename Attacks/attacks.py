import logging
from collections import OrderedDict

import torch


from Attacks.att_cba import AttCBA

from Attacks.att_neurotoxin import AttNeurotoxin
from Attacks.att_no_attack import NoAtt
from Attacks.att_no_attack_cdp import NoAttCDP
from Attacks.att_no_attack_ldp import NoAttLDP
from Attacks.att_dba import AttDBA
from Functions.log import get_logger

class Attacks:
    def __init__(self, cfg, AttClientId, dataset=None, identity='NoAtt'):
        self.cfg = cfg
        self.AttClientId = AttClientId
        self.identity = identity
        self.dataset = dataset

        self.method = self.get_attack_method(identity)



    def get_attack_method(self, attack='NoAtt'):
        print(f"|---Attack initing, Client [{self.AttClientId}] attack method is [{attack}]")
        if attack == 'NoAtt' or attack == "noatt":
            method = NoAtt(self.cfg)
        elif attack == 'NoAttLDP':
            method = NoAttLDP(self.cfg)
        elif attack == 'NoAttCDP':
            method = NoAttCDP(self.cfg)
        elif attack == 'NoAttCDP':
            method = NoAttCDP(self.cfg)
        elif attack == 'NoAttCDP':
            method = NoAttCDP(self.cfg)
        elif attack == 'Fang':
            method = AttackFang(self.cfg)
        elif attack == 'LIE':
            method = AttLie(self.cfg)
        elif attack == 'MinMax' or attack == 'MinSum':
            method = AttAdaptive(self.cfg)
        elif attack == 'SignFlipping':
            method = AttSignFlipping(self.cfg)
        elif attack == 'AdditiveNoise':
            method = AttAdditiveNoise(self.cfg)
        elif attack == 'CBA':
            method = AttCBA(self.cfg, self.AttClientId)
        elif attack == 'DBA':
            method = AttDBA(self.cfg, self.AttClientId)
        elif attack == 'Neurotoxin':
            method = AttNeurotoxin(self.cfg, self.AttClientId)
        elif attack == 'ORIDBA':
            method = AttOriDBA(self.cfg, self.AttClientId)
        elif attack == 'F3BA':
            method = AttF3BA(self.cfg, MalID = self.AttClientId, dataset=self.dataset)
        elif attack == 'Composite':
            method = AttComposite(self.cfg, MalID = self.AttClientId)
        else:
            print(f"Attack init fail. Your Attack method is {attack}")
            assert (2 == 1)
        return method

    def attack_model(self, e=None, pre_global_model=None, clients=None, choice_id=None,
                     malicious_id=None):

        self.method.injection(pre_global_model, clients, choice_id, malicious_id)
