import copy

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from Functions.log import get_logger
from Metrics.infosave import InfoSave
from Metrics.metric import Metric
from Metrics.metrics_crfl import MetricCRFL


class Metrics:
    def __init__(self, conf, train_dataset, test_dataset, attack=None):
        self.conf = conf
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=256, shuffle=False)

        self.attack = attack
        self.logger = get_logger(conf['logger']['logger_name'])
        self.info = InfoSave(conf)
        self.loadPoisoningData()

    def loadPoisoningData(self):
        self.train_dataset_poison = copy.deepcopy(self.train_dataset)
        self.test_dataset_poison_ = copy.deepcopy(self.test_dataset)
        self.test_dataset_poison = []
        for i, v in enumerate(self.test_dataset_poison_):
            if v[1] != self.conf['MalSetting']['BackdoorLabel']:
                self.test_dataset_poison.append(v)
        self.poison_test_dataloader = DataLoader(self.test_dataset_poison, batch_size=128)

    def evaluate_accuracy(self, model, test_dataloader, is_backdoor=False, backdoor_label=None):
        DEVICE = self.conf['DEVICE']
        model = model.to(DEVICE)
        model.eval()
        total_loss = 0.0
        correct = 0
        datasize = 0
        with torch.no_grad():
            if is_backdoor == False:
                for batch_id, batch in enumerate(test_dataloader):
                    data, target = batch
                    data = data.to(DEVICE)
                    target = target.to(DEVICE)
                    # sum up batch loss
                    output = model(data)
                    pred = output.data.max(1)[1]
                    correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
                    total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
                    datasize += data.shape[0]
            else:
                for batch_id, batch in enumerate(test_dataloader):
                    inputs, labels = batch
                    inputs = inputs.to(DEVICE)
                    labels = labels.to(DEVICE)
                    poison_imgs, poison_labels = self.attack.method.injectTrigger2Imgs(inputs,
                                                                                       labels,
                                                                                       backdoor_label,
                                                                                       Test=True)
                    outputs = model(poison_imgs)
                    pred = outputs.data.max(1)[1]
                    poison_labels.to(DEVICE)
                    pred = pred.to(DEVICE)
                    correct += pred.eq(poison_labels.view_as(pred)).cpu().sum().item()
                    total_loss += torch.nn.functional.cross_entropy(outputs, poison_labels,
                                                                    reduction='sum').item()
                    datasize += inputs.shape[0]
        # 一轮训练完后计算损失率和正确率
        loss = total_loss / datasize  # 当前轮的总体平均损失值
        acc = float(correct) / datasize  # 当前轮的总正确率
        return acc, loss, correct, datasize

    def evaluateClientEpoch(self, model, test_dataloader, id=None, e=None):
        conf = self.conf
        acc, loss, correct, datasize = self.evaluate_accuracy(model, test_dataloader,
                                                              is_backdoor=False)
        asr, loss_asr, asr_correct, asr_datasize = self.evaluate_accuracy(model, test_dataloader, is_backdoor=True,
                                                                          backdoor_label=conf['MalSetting'][
                                                                              'BackdoorLabel'])
        self.logger.info(f'|---Client {id} , Loss: {loss:.6f},  Acc:  {acc * 100:.4f}% ({correct}/{datasize})')
        self.logger.info(
            f'|---Client {id} , Loss: {loss_asr:.6f}, Asr: {asr * 100:.4f}% ({asr_correct}/{asr_datasize})')

    def evaluateEpoch(self, model, test_dataloader, poison_test_dataloader, e=None, clients=None, MalIDs=None,
                      backdoor_label=None):
        model = copy.deepcopy(model)
        conf = self.conf

        if self.conf['defense'] == 'CRFL':
            metric = MetricCRFL(conf, self.train_dataset, self.test_dataset, info=self.info,attack=self.attack)

            metric.evaluateCRFL(model, test_dataloader, poison_test_dataloader, e=e, clients=clients, MalIDs=MalIDs,
                      backdoor_label=backdoor_label)
        else:
            metric = Metric(conf, self.train_dataset, self.test_dataset, info=self.info, attack=self.attack)
            metric.evautePerformance(model, test_dataloader, poison_test_dataloader, e=e, clients=clients, MalIDs=MalIDs,
                      backdoor_label=backdoor_label)

