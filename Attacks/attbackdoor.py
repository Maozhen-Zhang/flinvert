import copy
import statistics
from collections import OrderedDict

import torch
import yaml

from Functions.helper import Helper
from Functions.log import get_logger


class AttBackdoor():
    def __init__(self, conf):
        self.conf = conf
        self.setBound()
        self.setNormalize(conf.img_size)

    def setEpochInfo(self, e):
        self.current_epoch = e
    def setBound(self):
        DEVICE = self.conf['DEVICE']
        dataset = self.conf['dataset']

        if dataset in ['mnist', 'fashion-mnist']:
            BoundShape = (1)
        elif dataset in ['cifar10','cifar100','tiny-imagenet']:
            BoundShape = (3)
        else:
            assert (2 == 1)
        upperbound = torch.ones(BoundShape).to(DEVICE)
        lowerbound = torch.zeros(BoundShape).to(DEVICE)

        if self.conf['Normalize'] == False:
            self.UpperBound = upperbound
            self.LowerBound = lowerbound
        else:
            if dataset == 'mnist':
                mean = [0.5]
                std = [0.5]
            elif dataset == 'cifar10':
                mean = [0.4914, 0.4822, 0.4465]
                std = [0.2023, 0.1994, 0.2010]
            elif dataset == 'tiny-imagenet':
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
            else:
                assert (2 == 1)
            t_mean = torch.FloatTensor(mean).to(DEVICE)
            t_std = torch.FloatTensor(std).to(DEVICE)
            self.UpperBound = (upperbound - t_mean) / t_std
            self.LowerBound = (lowerbound - t_mean) / t_std

    def setNormalize(self, img_size):
        conf = self.conf
        DEVICE = conf['DEVICE']
        channel = img_size
        self.mean = conf['mean']
        self.std = conf['std']
        self.t_mean = torch.FloatTensor(self.mean).view(channel, 1, 1).expand((3,img_size,img_size)).to(DEVICE)
        self.t_std = torch.FloatTensor(self.std).view(channel, 1, 1).expand((3,img_size,img_size)).to(DEVICE)

    def setL2Norm(self, l2norm):
        self.l2norm = l2norm

    def getL2Norm(self):
        return self.l2norm

    # use for DP
    def compute_local_model_update(self, PreModel, LastModel):
        if len(list(PreModel.parameters())) != len(list(LastModel.parameters())):
            raise AssertionError("Two models have different length in number of clients")
        norm_values = []
        for i in range(len(list(PreModel.parameters()))):
            norm_values.append(torch.norm(list(PreModel.parameters())[i] - list(LastModel.parameters())[i]))
        return statistics.median(norm_values)


    def compute_local_model_update_(self, PreModel, LastModel):
        if len(list(PreModel.parameters())) != len(list(LastModel.parameters())):
            raise AssertionError("Two models have different length in number of clients")
        norm_values = []
        for i in range(len(list(PreModel.parameters()))):
            norm_values.append((list(LastModel.parameters())[i] - list(PreModel.parameters())[i]))
        norm_flatten = torch.cat([torch.flatten(i) for i in norm_values])
        return torch.norm(norm_flatten).item()



    def tensor2Normalize(self, img):
        t_mean = self.t_mean
        t_std = self.t_std
        img = (img - t_mean) / t_std
        return img

    def tensor2Denormalize(self, img):
        t_mean = self.t_mean
        t_std = self.t_std
        img = img * t_std + t_mean
        return img

    def initTrigger(self):
        # TODO implement this
        return None

    def initMask(self):
        # TODO implement this
        return None

    def initPattern(self):
        # TODO implement this
        return None

    def setTrigger(self, mask, pattern):
        self.mask = mask
        self.pattern = pattern

    def getTrigger(self):
        return self.mask, self.pattern

    def injectTrigger(self, pattern=None):
        # TODO implement this
        return None

    def aggregate_weight(self, model, clients, chosen_ids, pts=None):
        assert not clients is None and len(clients) > 0
        averaged_weights = OrderedDict()
        for layer, weight in model.state_dict().items():
            averaged_weights[layer] = torch.zeros_like(weight).to(self.conf['device'])

        if pts is None:
            pts = [1 for i in range(len(chosen_ids))]
        total_prop = 0
        for pt, id in zip(pts, chosen_ids):
            client = clients[id]
            total_prop = total_prop + client.n_sample * pt
        for pt, id in zip(pts, chosen_ids):
            client = clients[id]
            # 每个用户数据集的数量
            prop = client.n_sample * (pt / total_prop)
            self.add_weights(averaged_weights, client.local_model.state_dict(), prop)
        return averaged_weights

    def add_weights(self, averaged_weights: OrderedDict, client_weights: OrderedDict, ratio):
        for layer in client_weights.keys():
            averaged_weights[layer] = averaged_weights[layer] + client_weights[layer] * ratio

    def avgGrad(self, grads):
        clientIDs = list(grads.keys())
        layers = grads[clientIDs[0]].keys()
        avg_grads = OrderedDict()
        for layer, weight in grads[clientIDs[0]].items():
            avg_grads[layer] = torch.zeros_like(weight).to(self.conf['device'])
        lam = {i: 1 / len(clientIDs) for i in clientIDs}
        for layer in layers:
            for i in clientIDs:
                avg_grads[layer] += grads[i][layer] * lam[i]
        return avg_grads

    def getWeightGrad(self, Param1, Param2):
        grad = OrderedDict()
        for layer in Param1.keys():
            if 'weight' in layer or 'bias' in layer:
                grad[layer] = Param1[layer] - Param2[layer]
            else:
                grad[layer] -= grad[layer]
        return grad

    def dict2Cuda(self, dict_, DEVICE):
        for key, values in dict_.items():
            values = values.to(DEVICE)
            dict_[key] = values
        return dict_

    def getPoissonDataset(self, train_dataset, pattern=None):
        poison_dataset = copy.deepcopy(train_dataset)
        return poison_dataset

    def injectTrigger2Imgs(self, imgs, labels, mask=None,pattern=None, target_label=None, Test=False):
        target_label = self.conf['MalSetting']['BackdoorLabel'] if target_label == None else target_label
        if Test:
            PoisonProportion = imgs.shape[0]
        else:
            PoisonProportion = int(
                (self.conf['MalSetting']['PoisonProportion'] / self.conf['batch_size']) * imgs.shape[0])
        if mask is None or pattern is None:
            mask = self.mask
            pattern = self.pattern
        poison_imgs = copy.deepcopy(imgs)
        poison_labels = copy.deepcopy(labels)
        # print((self.mask * self.pattern)[0,3:6,23:26])
        poison_imgs[:PoisonProportion] = (1 - mask) * poison_imgs[:PoisonProportion] + mask * pattern
        poison_labels[:PoisonProportion] = poison_labels[:PoisonProportion].fill_(target_label)
        return poison_imgs, poison_labels

    def train(self, epoch, model, trainloader, optimizer, criterion, DEVICE):
        conf = self.conf
        ScaleLambda = conf[conf['attack']]['ScaleLambda']
        SimLambda = conf[conf['attack']]['SimLambda']
        raw_model = copy.deepcopy(model)
        for e in range(epoch):
            for batch_idx, (data, target) in enumerate(trainloader):
                if data.shape[0] == 1:
                    data = torch.concat([data, data])
                    target = torch.concat([target, target])
                data, target = data.to(DEVICE), target.to(DEVICE)
                poison_data, poison_target = self.injectTrigger2Imgs(data, target)
                # print(poison_data[0,0,3:7,3:12]) if batch_idx == 0 else None
                # print(poison_data[0,0,3:6,3:10])

                optimizer.zero_grad()

                # model.zero_grad()
                output = model(data)
                loss1 = criterion(output, target).mean()

                # model.zero_grad()
                poison_output = model(poison_data)

                loss2 = criterion(poison_output, poison_target).mean()
                loss = ScaleLambda[0] * loss1 + ScaleLambda[1] * loss2
                loss += SimLambda * self.model_similarity_loss(raw_model, model)
                loss.backward()
                optimizer.step()

        # diff_norm = Helper.model_dist_norm(raw_mode, model.state_dict())
        # self.logger.info(f"|---更新后距离, {diff_norm}")

    def model_similarity_loss(self, global_model, local_model):
        # global_model.switch_grads(False)
        global_weights = global_model.state_dict()
        local_weights = local_model.state_dict()
        layers = global_weights.keys()
        loss = 0
        for layer in layers:
            if 'tracked' in layer or 'running' in layer:
                continue
            layer_dist = global_weights[layer] - local_weights[layer]
            loss = loss + torch.sum(layer_dist ** 2)
        return loss




#
# if __name__ == '__main__':
#     with open(f'../configs/conf__.yaml', "r+") as file:
#         conf = yaml.load(file, Loader=yaml.FullLoader)
#     conf['DEVICE'] = 'mps'
#     conf['ImageShape'] = (3, 32, 32)
#     att = AttackBackdoor(conf)
#     # print(att.UpperBound)
#     # print(att.LowerBound)
