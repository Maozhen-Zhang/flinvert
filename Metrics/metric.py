import copy

import torch
import wandb
from torch.utils.data import DataLoader

from Functions.log import get_logger


class Metric():
    def __init__(self, conf, train_dataset, test_dataset, info=None, attack=None):
        self.conf = conf
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=256, shuffle=False)

        self.attack = attack
        self.logger = get_logger(conf['logger']['logger_name'])
        self.info = info
        self.poison_test_dataloader = self.loadPoisoningData(self.conf['MalSetting']['BackdoorLabel'])
        self.poison_test_dataloader_target_1 = self.loadPoisoningData(1)
        self.poison_test_dataloader_target_0 = self.loadPoisoningData(0)

    def loadPoisoningData(self, BackdoorLabel):
        self.train_dataset_poison = copy.deepcopy(self.train_dataset)
        self.test_dataset_poison_ = copy.deepcopy(self.test_dataset)
        self.test_dataset_poison = []
        for i, v in enumerate(self.test_dataset_poison_):
            # self.conf['MalSetting']['BackdoorLabel']
            if v[1] != BackdoorLabel:
                self.test_dataset_poison.append(v)

        posion_test_dataloader = DataLoader(self.test_dataset_poison, batch_size=128)
        return posion_test_dataloader

    def evaluate_accuracy(self, model, test_dataloader, is_backdoor=False, injectTrigger2Imgs=None, mask=None,
                          pattern=None, backdoor_label=None):
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

                    poison_imgs, poison_labels = injectTrigger2Imgs(inputs,
                                                                    labels,
                                                                    mask=mask,
                                                                    pattern=pattern,
                                                                    target_label=backdoor_label,
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

    def evaluate_accuracyWithFilter(self, model, test_dataloader, is_backdoor=False, injectTrigger2Imgs=None,
                                    injectFilterTrigger2Imgs=None, mask=None,
                                    pattern=None, filter=None, backdoor_label=None):
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
                    if injectTrigger2Imgs != None:
                        inputs, labels = injectTrigger2Imgs(inputs,
                                                            labels,
                                                            mask=mask,
                                                            pattern=pattern,
                                                            target_label=backdoor_label,
                                                            Test=True)
                    if injectFilterTrigger2Imgs != None:
                        inputs, labels = injectFilterTrigger2Imgs(inputs, labels,
                                                                  mask=self.attack.method.filtertrigger,
                                                                  target_label=backdoor_label,
                                                                  Test=True)
                    outputs = model(inputs)
                    pred = outputs.data.max(1)[1]
                    labels.to(DEVICE)
                    pred = pred.to(DEVICE)
                    correct += pred.eq(labels.view_as(pred)).cpu().sum().item()
                    total_loss += torch.nn.functional.cross_entropy(outputs, labels,
                                                                    reduction='sum').item()
                    datasize += inputs.shape[0]
        # 一轮训练完后计算损失率和正确率
        loss = total_loss / datasize  # 当前轮的总体平均损失值
        acc = float(correct) / datasize  # 当前轮的总正确率
        return acc, loss, correct, datasize

    def aggMetricTrigger(self, clients, MalIDs):
        if self.attack.identity in self.conf['MalSetting']['BackdoorMethods']:
            self.attack.method.pattern = 0
            for i in MalIDs:
                self.attack.method.pattern += clients[i].attack.method.pattern
            self.attack.method.pattern = self.attack.method.pattern / len(MalIDs)


    def evautePerformance(self, model, test_dataloader, poison_test_dataloader, e=None, clients=None, MalIDs=None,
                          backdoor_label=None):
        model = copy.deepcopy(model)
        conf = self.conf

        acc, loss, correct, datasize = self.evaluate_accuracy(model, test_dataloader, is_backdoor=False)
        self.info.AccList.append(acc)
        self.info.LossList.append(loss)

        wandb.log({'acc': acc,'ACC': acc, 'loss': loss, 'epoch': e, 'Epoch': e}) if conf['wandb']['is_wandb'] else None
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
                if self.attack.method.TrainType == 'adaptive-multi-backdoor':
                    backdoor_label_1 = 1
                    backdoor_label_0 = 0
                    poison_test_dataloader_1 = self.poison_test_dataloader_target_1
                    poison_test_dataloader_0 = self.poison_test_dataloader_target_0
                else:
                    backdoor_label = self.conf['MalSetting']['BackdoorLabel']
                    backdoor_label_0 = backdoor_label
                    backdoor_label_1 = backdoor_label
                    poison_test_dataloader_1 = poison_test_dataloader
                    poison_test_dataloader_0 = poison_test_dataloader
                self.aggMetricTrigger(clients, MalIDs)

                pattern = self.attack.method.pattern
                maskA = self.attack.method.maskA
                AsrA, lossA, correct, datasize = self.evaluate_accuracy(model, poison_test_dataloader_1,
                                                                        injectTrigger2Imgs=injectTrigger2Imgs,
                                                                        mask=maskA,
                                                                        pattern=pattern,
                                                                        backdoor_label=backdoor_label_1,
                                                                        is_backdoor=True)
                self.logger.info(f'|---Epoch: {e}, backdoor label is {backdoor_label_1}')
                self.logger.info(f'|---Epoch: {e}, Loss: {lossA:.6f},  AccA: {AsrA * 100:.4f}% ({correct}/{datasize})')
                wandb.log({'AsrA': AsrA, 'lossA': lossA, 'epoch': e}) if conf['wandb']['is_wandb'] else None

                maskB = self.attack.method.maskB
                AsrB, lossB, correct, datasize = self.evaluate_accuracy(model, poison_test_dataloader_0,
                                                                        injectTrigger2Imgs=injectTrigger2Imgs,
                                                                        mask=maskB,
                                                                        pattern=pattern,
                                                                        backdoor_label=backdoor_label_0,
                                                                        is_backdoor=True)
                self.logger.info(f'|---Epoch: {e}, backdoor label is {backdoor_label_0}')
                self.logger.info(f'|---Epoch: {e}, Loss: {lossB:.6f},  AccB: {AsrB * 100:.4f}% ({correct}/{datasize})')
                wandb.log({'AsrB': AsrB, 'lossB': lossB, 'epoch': e}) if conf['wandb']['is_wandb'] else None
