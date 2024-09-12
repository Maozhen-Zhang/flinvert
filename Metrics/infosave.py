import csv
import os.path
import pickle

import numpy as np
import torch
import yaml

from Functions.log import get_logger
from Functions.process_image import saveTensor2Img, tensor2denormalize


class InfoSave:
    def __init__(self, conf):
        self.conf = conf
        self.logger = get_logger(conf['logger']['logger_name'])
        self.info = {}
        self.AccList = []
        self.AsrList = []
        self.LossList = []
        self.AsrLossList = []
        self.initRootPath()
        if conf['Save']['IsSave']:
            self.initSavePath()

    def initRootPath(self):
        self.RootPath = self.conf['Save']['RootPath']
        if not os.path.exists(self.RootPath):
            os.makedirs(self.RootPath)
        self.RootInfoPath = self.conf['Save']['RootInfoPath']
        if not os.path.exists(self.RootInfoPath):
            os.makedirs(self.RootInfoPath)

    def initSavePath(self):
        conf = self.conf
        self.conf['Save']['SavePath'] = conf['Save']['SavePath'] + conf['dataset'] + '-' + conf['model'] + '-' + conf[
            'attack'] + '-' + conf['defense'] + '-NonIID-' + str(conf['heterogenuity']['dirichlet_alpha'])

        self.SavePath = os.path.join(self.RootPath + self.conf['Save']['SavePath'])
        if not os.path.exists(self.SavePath):
            os.makedirs(self.SavePath)
        self.logger.info(f"|---SavePath is {self.SavePath}")

        self.SaveInfoPath = os.path.join(self.RootInfoPath, self.conf['Save']['SavePath'])
        if not os.path.exists(self.SaveInfoPath):
            os.makedirs(self.SaveInfoPath)
        self.logger.info(f"|---SaveInfoPath is {self.SaveInfoPath}")

    def saveParam(self):
        with open(f'{self.SavePath}/params.yaml', 'w') as f:
            yaml.dump(self.conf, f)

    def saveEvaluateCSV(self):
        if self.conf['Save']['IsSave']:
            EvaluateSaveName = self.conf['Save']['AccSaveName']
            SavePath = os.path.join(self.SavePath, EvaluateSaveName)
            with open(SavePath, 'w', newline='') as f:
                writer = csv.writer(f)
                if self.conf['attack'] in self.conf['MalSetting']['BackdoorMethods']:
                    writer.writerow(['epoch', 'acc', 'asr', 'loss', 'asrloss'])
                else:
                    writer.writerow(['epoch', 'acc', 'loss'])
                for i in range(len(self.AccList)):
                    if self.conf['attack'] in self.conf['MalSetting']['BackdoorMethods']:
                        writer.writerow([i, self.AccList[i], self.AsrList[i], self.LossList[i], self.AsrLossList[i]])
                    else:
                        writer.writerow([i, self.AccList[i], self.LossList[i]])

            self.logger.info(f"|---Evaluation Result save success，path is {SavePath}")
            if self.conf['Save']['IsSaveModel'] and self.conf['Save']['IsSave']:
                self.logger.info(f"|---Model is  save success")
            if self.conf['Save']['IsSaveTrigger'] and self.conf['Save']['IsSave']:
                self.logger.info(f"|---Trigger is  save success")

    def saveClientWeight(self, clients, choice_id, e):
        if self.conf['Save']['IsSaveModel'] and self.conf['Save']['IsSave']:
            saveDict = {}
            saveDict['epoch'] = e
            for i in choice_id:
                client = clients[i]
                saveDict[i] = client.local_model.state_dict()
            ClientWeightsFileName = 'Epoch-' + str(e) + '-Clients.pkl'
            SavePath = os.path.join(self.SaveInfoPath, ClientWeightsFileName)
            with open(SavePath, 'wb') as file:
                pickle.dump(saveDict, file)

    def saveModelWeight(self, model, e):
        if self.conf['Save']['IsSaveModel'] and self.conf['Save']['IsSave']:
            saveDict = {}
            saveDict['epoch'] = e
            saveDict['global'] = model.state_dict()
            GlobalWeightsFileName = 'Epoch-' + str(e) + '-Global.pkl'
            SavePath = os.path.join(self.SaveInfoPath, GlobalWeightsFileName)
            with open(SavePath, 'wb') as file:
                pickle.dump(saveDict, file)
            # self.logger.info(f"|---Model is  save success，path is {SavePath}")

    def saveTrigger(self, e, mask, pattern):
        if self.conf['Save']['IsSaveTrigger'] and self.conf['Save']['IsSave'] and self.conf['attack'] in self.conf['MalSetting']['BackdoorMethods']:
            trigger = {"mask": mask, "pattern": pattern}
            TriggerSaveName = 'Epoch-' + str(e) + '-Trigger.pkl'
            SavePath = os.path.join(self.SaveInfoPath, TriggerSaveName)
            with open(SavePath, 'wb') as file:
                pickle.dump(trigger, file)

            TriggerSaveName = 'Epoch-' + str(e) + '-Trigger.png'
            SavePath = os.path.join(self.SaveInfoPath, TriggerSaveName)
            trigger = mask * pattern
            # trigger = trigger.cpu().numpy()
            # print()
            if self.conf['Normalize'] == True:
                trigger = tensor2denormalize(trigger)
            trigger = trigger.cpu().numpy()
            mask = mask.cpu().numpy()
            for channel in range(trigger.shape[0]):
                trigger[channel][np.where(mask == 0)] = 0.5
            trigger = torch.from_numpy(trigger)
            saveTensor2Img(trigger, SavePath)
