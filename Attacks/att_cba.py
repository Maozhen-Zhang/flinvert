import copy

import numpy as np
import torch
import yaml

from Attacks.attbackdoor import AttBackdoor
from Functions.log import get_logger, configure_logger

"""
    How To Backdoor Federated Learning
"""
class AttCBA(AttBackdoor):
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
            self.logger.debug(f"|---Total Mask init is---:\n" + str(self.Totalmask[3:6, 23:26]))
            # self.logger.debug(f"|---Total trigger init is---:\n" + str((self.mask * self.pattern)[:, 3:7, 3:13]))

        if self.MalID in self.MalIDs:
            self.logger.debug(f"|---MalID  is---{self.MalID}")
            self.logger.debug(f"|---local Mask init is---:\n" + str(self.mask[3:6, 23:26]))
            self.logger.debug(f"|---Local trigger init is---:\n" + str((self.mask * self.pattern)[:, 3:6, 23:26]))

    def initMask(self):
        DEVICE = self.conf['DEVICE']

        self.TotalPoisonLocation = [
            [3, 23], [3, 24], [3, 25],
            [4, 23], [4, 24], [4, 25],
            [5, 23], [5, 24], [5, 25],
        ]
        self.PoisonLocation = self.TotalPoisonLocation
        self.Totalmask = torch.zeros(self.conf['ImageShape'][-2:]).to(DEVICE)
        for xloc, yloc in self.TotalPoisonLocation:
            self.Totalmask[xloc, yloc] = 1

        self.mask = torch.zeros(self.conf['ImageShape'][-2:]).to(DEVICE)
        if self.MalID != None and self.MalID in self.MalIDs:
            PoisonLocation = self.PoisonLocation
            for poison_x, poison_y in PoisonLocation:
                self.mask[poison_x, poison_y] = 1
        elif self.MalID == -1:
            self.mask = self.Totalmask

    def initPattern(self):
        DEVICE = self.conf['DEVICE']
        dataset = self.conf['dataset']

        if dataset == 'mnist' or dataset == 'cifar10'or dataset == 'cifar100' or dataset == 'tiny-imagenet':
            self.pattern = torch.ones(self.conf['ImageShape']).to(DEVICE)
            PatternValues = self.UpperBound
            for channel in range(len(PatternValues)):
                self.pattern[channel].fill_(PatternValues[channel]).to(DEVICE)
        # self.pattern = self.getNormalize(self.pattern)


if __name__ == '__main__':
    with open(f'../configs/conf.yaml', "r+") as file:
        conf = yaml.load(file, Loader=yaml.FullLoader)
    configure_logger(conf)
    conf['DEVICE'] = 'mps'
    conf['ImageShape'] = (3, 32, 32)
    att = AttCBA(conf, -1)
    att.initTrigger()
