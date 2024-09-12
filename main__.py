# 查看当前解释器
# import sys
#
# module_path = ['Aggregation','Attacks','configs','Functions','Metrics','Terminial']
# for i in range(len(module_path)):
#     sys.path.append(module_path[i])
# print(sys.path)
import copy
import logging
import os
import pickle

# Your code that generates warnings here
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from Aggregation.defense import Defense
from Attacks.attacks import Attacks
from Functions.conf import getConf
import wandb

from Functions.data import get_dataset, get_data_indicates, print_distribution
from Functions.get_model import init_model
from Functions.helper import Helper
from Functions.log import get_logger, configure_logger
from Metrics.metrics import Metrics
from Terminal.client import Client
from Terminal.server import Server


# from Test.FLTest import evaluate_accuracy, loadPoisoningData
logger = logging.getLogger('PrivacyEngine')
# 设置记录器的级别为CRITICAL，以关闭日志输出
logger.setLevel(logging.CRITICAL)


def initGlobal():

    conf = getConf()
    # configure_logger(conf)
    logger = get_logger(configure_logger(conf))

    # --- dataset init---
    logger.info(f"|---dataset init---")
    train_dataset, test_dataset = get_dataset(dataset=conf['dataset'], root_path=conf['root_path'],
                                              IsNormalize=conf['Normalize'])
    logger.info(f"|---slice init---")
    dataset_slices = get_data_indicates(conf, train_dataset, test_dataset)
    # print_distribution(conf, dataset_slices)  # print distribution
    logger.info(f"|---MalIDs init---")

    Nclient = conf['n_client']
    Nmal = conf['n_mal']
    TotalClientID = [i for i in range(Nclient)]
    MalIDs = conf['MalSetting']['MalIDs']


    GlobalAttack = Attacks(conf, -1, identity=conf['attack'])

    logger.info(f"|---Metric init---")
    metric = Metrics(conf, train_dataset, test_dataset, attack=GlobalAttack)
    Helper.setMetric(metric)
    #
    model = init_model(conf)
    logger.info(f"|---terminal init---")

    # server register
    server = Server(conf, model, metric, Defense(conf, metric=metric))
    # client register
    clients = {}
    for ClientID in TotalClientID:
        model = init_model(conf)
        identity = conf['attack'] if ClientID in MalIDs else conf['benign']
        attack = Attacks(conf, ClientID, dataset=dataset_slices[ClientID], identity=identity)

        client = Client(conf, ClientID, dataset_slices[ClientID], test_dataset, model, attack=attack,
                        metric=metric)
        clients[ClientID] = client

    SampleClientID = conf['sample_client']
    epoch = [i for i in range(conf['epoch'])]
    if conf['Resume']['IsResume']:
        BeginEpoch = conf['Resume']['BeginEpoch']
        epoch = [i for i in range(BeginEpoch + 1, BeginEpoch + 1)]
        ReloadModelPath = conf['Resume']['ResumeModelPath'] + 'Epoch-' + str(
            conf['Resume']['BeginEpoch']) + '-Global.pkl'
        logger.info(f"|---Reload Model Path is {ReloadModelPath}")
        with open(ReloadModelPath, 'rb') as f:
            ReloadModelWeight = pickle.load(f)

        server.global_model.load_state_dict(ReloadModelWeight['global'])

        # epoch = [i for i in range(BeginEpoch + 1, conf['epoch'])]
        # ReloadModelPath = conf['Resume']['ResumeModelPath']
        # ReloadModelWeight = torch.load(f"{ReloadModelPath}", map_location="cpu")
        # server.global_model.load_state_dict(ReloadModelWeight['state_dict'])
        server.global_model.to(conf['DEVICE'])

    return TotalClientID, SampleClientID, epoch, server, clients, GlobalAttack, metric, logger, conf


def Train(TotalClientID, SampleClientID, epoch, server, clients, attack, metric, logger, conf):
    for e in epoch:
        logger.info(f'|---' + '=' * 50)
        logger.info(f'|---Epoch: [{e}]')
        logger.info(f'|------Dataset is [{conf["dataset"]}], Model is [{conf["model"]}]')

        logger.info(f'|------Agg is [{conf["defense"]}], Att is [{conf["attack"]}]')
        logger.info(f'|------Lr is [{conf["lr"]}]')
        if conf['attack'] in conf['MalSetting']['BackdoorMethods']:
            logger.info(f'|------Poison Lr is [{conf[conf["attack"]]["poison_lr"]}]')
            logger.info(f'|------Poison proportion is [{conf["MalSetting"]["PoisonProportion"]}]')
        choice_id, malIDs = server.getChoiceId(TotalClientID, SampleClientID)

        logger.info(f"|---Total client id is {choice_id}")
        logger.info(f"|---Malic client id is {malIDs}")

        server.broad_global_weight(clients, choice_id)

        server.defense.setEpochInfo(e)
        for i in choice_id:
            client = clients[i]
            client.setEpochInfo(e)
            client.train()
            if conf['Print'][
                'PrintClientAcc'] == True and conf['attack'] in conf['MalSetting']['BackdoorMethods'] and i in malIDs and client.attack.identity != 'NoAtt':
                metric.evaluateClientEpoch(copy.deepcopy(client.local_model), client.trainloader, id=client.client_id,
                                           e=e)
        if conf['attack'] in conf['MalSetting']['RobustMethods']:
            attack.attack_model(e=e, pre_global_model=server.global_model, clients=clients, choice_id=choice_id,
                                malicious_id=malIDs)
        server.aggregate_weight(clients, choice_id)
        metric.evaluateEpoch(copy.deepcopy(server.global_model), metric.test_dataloader, metric.poison_test_dataloader,
                             e,
                             clients=clients,
                             MalIDs=conf['MalSetting']['MalIDs'],
                             backdoor_label=conf['MalSetting']['BackdoorLabel'])
        metric.info.saveClientWeight(clients, choice_id, e)
        metric.info.saveModelWeight(server.global_model, e)
        metric.info.saveTrigger(e, metric.attack.method.mask, metric.attack.method.pattern)
    metric.info.saveEvaluateCSV()


def OneShotDefense():

    choice_id, malIDs = server.getChoiceId(TotalClientID, SampleClientID)

    CurrentEpoch = conf['Resume']['BeginEpoch'] + 1
    server.defense.setEpochInfo(CurrentEpoch)
    server.broad_global_weight(clients, choice_id)
    PreGlobalmodel = copy.deepcopy(server.global_model)
    for i in choice_id:
        client = clients[i]
        client.local_model.to(conf['DEVICE'])

    server.aggregate_weight(clients, choice_id)

    metric.evaluateEpoch(PreGlobalmodel, metric.test_dataloader, metric.poison_test_dataloader, CurrentEpoch - 1,
                         clients=clients,
                         MalIDs=conf['MalSetting']['MalIDs'],
                         backdoor_label=conf['MalSetting']['BackdoorLabel'])

    metric.evaluateEpoch(server.global_model, metric.test_dataloader, metric.poison_test_dataloader, CurrentEpoch,
                         clients=clients,
                         MalIDs=conf['MalSetting']['MalIDs'],
                         backdoor_label=conf['MalSetting']['BackdoorLabel'])
    acc = metric.info.AccList[-1]
    asr = metric.info.AsrList[-1]

    with open('example.txt', 'a+') as file:
        # 写入新的一行数据
        file.write('\n')  # 在新的一行写入
        file.write(
            conf['dataset'] + '\t' +
            conf['FedOSD']['Defense'] + '\t' +
            'Acc' + ':' + str(acc) + '\t' +
            'ASR' + ':' + str(asr) + '\t'
                   )  # 每个元素写入新的一行
if __name__ == '__main__':

    TotalClientID, SampleClientID, epoch, server, clients, attack, metric, logger, conf = initGlobal()
    Train(TotalClientID, SampleClientID, epoch, server, clients, attack, metric, logger, conf)
