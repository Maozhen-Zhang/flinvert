import copy

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from Functions.log import get_logger
from Metrics.infosave import InfoSave
from Metrics.metric import Metric


class MetricCRFL(Metric):
    def __init__(self, conf, train_dataset, test_dataset, info=None,attack=None):
        super(MetricCRFL, self).__init__(conf, train_dataset, test_dataset, info,attack=attack)

    def add_differential_privacy_noise(self, model, sigma=0.001, cp=False):
        if not cp:
            for name, param in model.state_dict().items():
                if 'tracked' in name or 'running' in name:
                    continue
                # print(name)
                dp_noise = torch.cuda.FloatTensor(param.shape).normal_(mean=0, std=sigma)
                param.add_(dp_noise)
        else:
            smoothed_model = copy.deepcopy(model)
            for name, param in smoothed_model.state_dict().items():
                if 'tracked' in name or 'running' in name:
                    continue
                dp_noise = torch.cuda.FloatTensor(param.shape).normal_(mean=0, std=sigma)
                param.add_(dp_noise)
            return smoothed_model

    def evaluate_accuracy(self, model, test_dataloader, is_backdoor=False, injectTrigger2Imgs=None, mask=None,
                          pattern=None, backdoor_label=None):
        target_model = copy.deepcopy(model)
        smoothed_models = [self.add_differential_privacy_noise(model, sigma=0.002, cp=True) for m in range(5)]
        correct = 0
        total_loss = 0
        datasize = 0
        with torch.no_grad():
            for i, data in enumerate(test_dataloader):
                inputs, labels = data
                inputs = inputs.to(self.conf['DEVICE'])
                labels = labels.to(self.conf['DEVICE'])
                datasize += inputs.size(0)
                if is_backdoor:
                    inputs, labels = injectTrigger2Imgs(inputs,
                                                        labels,
                                                        mask=mask,
                                                        pattern=pattern,
                                                        target_label=backdoor_label,
                                                        Test=True)
                outputs = 0
                for m in range(len(smoothed_models)):
                    prob = torch.nn.functional.softmax(smoothed_models[m](inputs), 1)
                    outputs = outputs + prob
                outputs = outputs / (len(smoothed_models))
                _, pred = torch.max(outputs, 1)
                correct += pred.eq(labels.view_as(pred)).cpu().sum().item()
                total_loss += torch.nn.functional.cross_entropy(outputs, labels, reduction='sum').item()

            # 一轮训练完后计算损失率和正确率
            loss = total_loss / datasize  # 当前轮的总体平均损失值
            acc = float(correct) / datasize  # 当前轮的总正确率
            return acc, loss, correct, datasize

    def evaluateCRFL(self, model, test_dataloader, poison_test_dataloader, e=None, clients=None,
                     MalIDs=None,
                     backdoor_label=None):
        model = copy.deepcopy(model)
        conf = self.conf

        acc, loss, correct, datasize = self.evaluate_accuracy(model, test_dataloader, is_backdoor=False)
        self.info.AccList.append(acc)
        self.info.LossList.append(loss)

        wandb.log({'acc': acc, 'loss': loss, 'epoch': e}) if conf['wandb']['is_wandb'] else None
        self.logger.info(f'|---Epoch: {e}, Loss: {loss:.6f},  Acc: {acc * 100:.4f}% ({correct}/{datasize})')

        if conf['attack'] != 'NoAtt' and conf['attack'] in self.conf['MalSetting']['BackdoorMethods']:

            self.aggMetricTrigger(clients, MalIDs)
            injectTrigger2Imgs = self.attack.method.injectTrigger2Imgs
            asr, loss_asr, correct, datasize = self.evaluate_accuracy(model, poison_test_dataloader,
                                                                      injectTrigger2Imgs=injectTrigger2Imgs,
                                                                      is_backdoor=True,
                                                                      backdoor_label=backdoor_label)
            self.logger.info(f'|---Epoch: {e}, Loss: {loss_asr:.6f},  Asr: {asr * 100:.4f}% ({correct}/{datasize})')

            self.info.AsrList.append(asr)
            self.info.AsrLossList.append(loss_asr)

            wandb.log({'asr': asr, 'loss_asr': loss_asr, 'epoch': e}) if conf['wandb']['is_wandb'] else None
            self.logger.info(f'|---' + '=' * 50)

            if self.conf['attack'] == 'Composite':
                self.aggMetricTrigger(clients, MalIDs)
                backdoor_label = self.conf['MalSetting']['BackdoorLabel']

                pattern = self.attack.method.pattern
                maskA = self.attack.method.maskA
                AsrA, lossA, correct, datasize = self.evaluate_accuracy(model, poison_test_dataloader,
                                                                        injectTrigger2Imgs=injectTrigger2Imgs,
                                                                        mask=maskA,
                                                                        pattern=pattern,
                                                                        is_backdoor=True)
                self.logger.info(f'|---Epoch: {e}, Loss: {lossA:.6f},  AccA: {AsrA * 100:.4f}% ({correct}/{datasize})')
                wandb.log({'AsrA': AsrA, 'lossA': lossA, 'epoch': e}) if conf['wandb']['is_wandb'] else None

                maskB = self.attack.method.maskB
                AsrB, lossB, correct, datasize = self.evaluate_accuracy(model, poison_test_dataloader,
                                                                        injectTrigger2Imgs=injectTrigger2Imgs,
                                                                        mask=maskB,
                                                                        pattern=pattern,
                                                                        is_backdoor=True)
                self.logger.info(f'|---Epoch: {e}, Loss: {lossB:.6f},  AccB: {AsrB * 100:.4f}% ({correct}/{datasize})')

                wandb.log({'AsrB': AsrB, 'lossB': lossB, 'epoch': e}) if conf['wandb']['is_wandb'] else None
