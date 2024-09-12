import copy
import numpy as np
import torch
import yaml

from Attacks.attbackdoor import AttBackdoor
from Functions.log import get_logger, configure_logger

"""
    DBA: Distributed Backdoor Attacks against Federated Learning
"""
class AttDBA(AttBackdoor):
    def __init__(self, conf, MalID=None):
        super().__init__(conf)
        self.conf = conf
        self.logger = get_logger(conf['logger']['logger_name'])
        self.setMalSettingPerRount(MalID)
        self.initTrigger()

    def setMalSettingPerRount(self, MalID=None):
        self.MalNumPerRound = self.conf['MalSetting']['MalNumPerRound']
        self.MalIDs = self.conf['MalSetting']['MalIDs']
        self.MalID = MalID


    def initTrigger(self):
        DEVICE = self.conf['DEVICE']
        self.initPattern()
        self.initMask()

        self.mask = self.mask.to(DEVICE)
        self.Totalmask = self.Totalmask.to(DEVICE)
        if self.MalID == -1:
            self.logger.debug(f"|---Total Mask init is---:\n" + str(self.Totalmask[3:7, 3:13]))
            # self.logger.debug(f"|---Total trigger init is---:\n" + str((self.mask * self.pattern)[:, 3:7, 3:13]))

        if self.MalID in self.MalIDs:
            self.logger.debug(f"|---MalID  is---{self.MalID}")
            self.logger.debug(f"|---local Mask init is---:\n" + str(self.mask[3:7, 3:13]))
            self.logger.debug(f"|---Local trigger init is---:\n" + str((self.mask * self.pattern)[:, 3:7, 3:13]))




    def initMask(self):
        DEVICE = self.conf['DEVICE']

        self.TotalPoisonLocation = [
            [3, 3], [3, 4], [3, 5], [3, 6],
            [3, 9], [3, 10], [3, 11], [3, 12],
            [6, 3], [6, 4], [6, 5], [6, 6],
            [6, 9], [6, 10], [6, 11], [6, 12],
        ]
        self.PoisonLocation = {
            0: [[3, 3], [3, 4], [3, 5], [3, 6]],
            1: [[3, 9], [3, 10], [3, 11], [3, 12]],
            2: [[6, 3], [6, 4], [6, 5], [6, 6]],
            3: [[6, 9], [6, 10], [6, 11], [6, 12]],
        }
        if self.conf['dataset'] == 'tiny-imagenet':
            self.TotalPoisonLocation = [
                [3, 3], [3, 4], [3, 5], [3, 6],[3 ,7], [3, 8],
                  [3, 11], [3, 12],[3 ,13], [3, 14],[3, 15],
                [6, 3], [6, 4], [6, 5], [6, 6],[6 ,7], [6, 8],
                  [6, 11], [6, 12],[6, 13], [6, 14],[6, 15]
            ]
            self.PoisonLocation = {
                0: [[3, 3], [3, 4], [3, 5], [3, 6],[3 ,7], [3, 8]],
                1: [[3, 11], [3, 12],[3 ,13], [3, 14],[3, 15]],
                2: [[6, 3], [6, 4], [6, 5], [6, 6],[6 ,7], [6, 8]],
                3: [[6, 11], [6, 12],[6, 13], [6, 14],[6, 15]],
            }
        self.Totalmask = torch.zeros(self.conf['ImageShape'][-2:]).to(DEVICE)
        for xloc, yloc in self.TotalPoisonLocation:
            self.Totalmask[xloc, yloc] = 1

        self.mask = torch.zeros(self.conf['ImageShape'][-2:]).to(DEVICE)
        if self.MalID != None and self.MalID in self.MalIDs:
            PoisonLocation = self.PoisonLocation[self.MalID]
            for poison_x, poison_y in PoisonLocation:
                self.mask[poison_x, poison_y] = 1
        elif self.MalID == -1:
            self.mask = self.Totalmask

    def initPattern(self):
        DEVICE = self.conf['DEVICE']
        dataset = self.conf['dataset']

        if dataset == 'mnist':
            self.pattern = torch.ones(self.conf['ImageShape']).to(DEVICE)
            PatternValues = self.UpperBound
            for channel in range(len(PatternValues)):
                self.pattern[channel].fill_(PatternValues[channel]).to(DEVICE)
        elif dataset == 'cifar10':
            self.pattern = torch.ones(self.conf['ImageShape']).to(DEVICE)
            PatternValues = self.UpperBound
            for channel in range(len(PatternValues)):
                self.pattern[channel].fill_(PatternValues[channel]).to(DEVICE)
        if dataset == 'cifar100' or dataset == 'tiny-imagenet':
            self.pattern = torch.ones(self.conf['ImageShape']).to(DEVICE)
            PatternValues = self.UpperBound
            for channel in range(len(PatternValues)):
                self.pattern[channel].fill_(PatternValues[channel]).to(DEVICE)

if __name__ == '__main__':
    with open(f'../configs/conf.yaml', "r+") as file:
        conf = yaml.load(file, Loader=yaml.FullLoader)
    configure_logger(conf)
    conf['DEVICE'] = 'mps'
    conf['ImageShape'] = (3, 32, 32)
    att = AttDBA(conf, -1)
    att.setTrigger()
