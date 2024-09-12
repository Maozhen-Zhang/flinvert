import copy
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from Attacks.attbackdoor import AttBackdoor
from Functions.data import Subset
from Functions.log import configure_logger, get_logger
from Models.Alex import alex_cifar10
from Models.extractor import FeatureExtractor
from Models.init_model import CNNMnist
from Models.model import Model
from Models.resnet import layer2module, ResNet
from Models.simple import SimpleNet


def get_conv_weight_names(model: Model):
    conv_targets = list()
    weights = model.state_dict()
    for k in weights.keys():
        if 'conv' in k and 'weight' in k:
            conv_targets.append(k)

    return conv_targets


def get_neuron_weight_names(model: Model):
    neuron_targets = list()
    weights = model.state_dict()
    for k in weights.keys():
        if 'fc' in k and 'weight' in k:
            neuron_targets.append(k)

    return neuron_targets

"""
    On the Vulnerability of Backdoor Defenses for Federated Learning
"""
class AttF3BA(AttBackdoor):
    def __init__(self, conf, MalID=None, dataset=None):
        super().__init__(conf)
        self.conf = conf
        self.logger = get_logger(conf['logger']['logger_name'])
        self.dataset = dataset
        self.setMalSettingPerRount(MalID)
        self.initTrigger()
        self.handcraft_trigger = True

        if self.dataset != None:
            data_idxs = range(len(self.dataset))
            nt = int(len(self.dataset) * 0.8)
            train_ids, test_ids = data_idxs[:nt], data_idxs[nt:]
            self.train_dataset = Subset(self.dataset, train_ids)
            self.handcraft_dataset = Subset(self.dataset, test_ids)
            self.train_loader = DataLoader(self.train_dataset, batch_size=conf['batch_size'], shuffle=True,
                                           drop_last=True)
            self.handcraft_loader = DataLoader(self.handcraft_dataset, batch_size=conf['batch_size'], shuffle=True,
                                               drop_last=True)

        self.handcraft_rnd = 0
        self.previous_global_model = None

    def setMalSettingPerRount(self, MalID=None):
        self.MalNumPerRound = self.conf['MalSetting']['MalNumPerRound']
        self.MalIDs = self.conf['MalSetting']['MalIDs']
        self.MalID = MalID

    def initTrigger(self):
        DEVICE = self.conf['DEVICE']
        self.initMask()
        self.initPattern()

        self.mask = self.mask.to(DEVICE)
        self.Totalmask = self.Totalmask.to(DEVICE)
        self.setTrigger(self.mask, self.pattern)
        if self.conf['Print']['PrintTriggerInfo']:
            if self.MalID == -1:
                self.logger.debug(f"|---Total Mask init is---:\n" + str(self.Totalmask[3:6, 23:26]))
                # self.logger.debug(f"|---Total trigger init is---:\n" + str((self.mask * self.pattern)[0, 3:7, 3:13]))

            if self.MalID in self.MalIDs:
                self.logger.debug(f"|---MalID  is---{self.MalID}")
                self.logger.debug(f"|---local Mask init is---:\n" + str(self.mask[3:6, 23:26]))
                self.logger.debug(f"|---Local trigger init is---:\n" + str((self.mask * self.pattern)[0, 3:6, 23:26]))


    def initMask(self):
        DEVICE = self.conf['DEVICE']
        self.trigger_size = (3, 3)
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
        if dataset == 'mnist' or dataset == 'cifar10':
            self.pattern = torch.ones(self.conf['ImageShape']).to(DEVICE)

            pattern_tensor = torch.rand(self.trigger_size)
            pattern_tensor = (pattern_tensor * 255).floor() / 255
            tmp_mask = np.array(copy.deepcopy(self.mask).cpu())
            tmp_pattern = copy.deepcopy(self.pattern).cpu()
            for channel in range(self.conf['ImageShape'][0]):
                tmp_pattern[channel][np.where(tmp_mask == 1)] = pattern_tensor.reshape(-1)
            self.pattern = tmp_pattern.to(DEVICE)
        # self.pattern = self.tensor2Normalize(self.pattern)

    def train(self, epoch, model, trainloader, optimizer, criterion, DEVICE):
        previous_global_model = copy.deepcopy(model)
        self.handcraft(model, self.handcraft_loader, self.train_loader, previous_global_model)
        super().train(epoch, model, trainloader, optimizer, criterion, DEVICE)

    def handcraft(self, model, handcraft_loader, train_loader, previous_global_model):
        self.handcraft_rnd = self.handcraft_rnd + 1
        model.eval()
        if self.previous_global_model is None:
            self.previous_global_model = copy.deepcopy(model)
            return
        candidate_weights = self.search_candidate_weights(model, proportion=0.1)
        self.previous_global_model = copy.deepcopy(model)

        # 使用修改后的模型参数optimizer trigger
        if self.handcraft_trigger:
            self.logger.debug("Optimize Trigger:")
            self.optimize_backdoor_trigger(model, candidate_weights, handcraft_loader)

        self.logger.debug("Inject Candidate Filters:")
        diff = self.inject_handcrafted_filters(model, candidate_weights, handcraft_loader)
        if diff is not None and self.handcraft_rnd % 3 == 1:
            self.logger.debug("Rnd {}: Inject Backdoor FC".format(self.handcraft_rnd))
            self.inject_handcrafted_neurons(model, candidate_weights, diff, handcraft_loader)

    def search_candidate_weights(self, model: Model, proportion=0.1):
        self.kernel_selection = 'movement'
        assert self.kernel_selection in ['random', 'movement']
        candidate_weights = OrderedDict()
        model_weights = model.state_dict()
        n_labels = 0

        if self.kernel_selection == "movement":
            history_weights = self.previous_global_model.state_dict()
            for layer in history_weights.keys():
                if 'conv' in layer:
                    conv_rate = 0.02
                    proportion = conv_rate
                elif 'fc' in layer:
                    fc_rate = 0.001
                    proportion = fc_rate
                model_weights[layer] = model_weights[layer].to(self.conf['DEVICE'])
                history_weights[layer] = history_weights[layer].to(self.conf['DEVICE'])

                # candidate_weights[layer] = (model_weights[layer] - history_weights[layer]) * model_weights[layer]
                candidate_weights[layer] = abs((model_weights[layer] - history_weights[layer]) * model_weights[layer])

                n_weight = candidate_weights[layer].numel()
                theta = torch.sort(candidate_weights[layer].flatten(), descending=False)[0][
                    int(n_weight * proportion)]
                candidate_weights[layer][candidate_weights[layer] < theta] = 1
                candidate_weights[layer][candidate_weights[layer] != 1] = 0

        return candidate_weights

    def trigger_loss(self, model, backdoor_inputs, clean_inputs, pattern, grads=True):
        model.train()
        backdoor_activations = model.first_activations(backdoor_inputs).mean([0, 1])
        clean_activations = model.first_activations(clean_inputs).mean([0, 1])
        difference = backdoor_activations - clean_activations
        loss = torch.sum(difference * difference)

        if grads:
            grads = torch.autograd.grad(loss, pattern, retain_graph=True)
        return loss, grads

    '''优化trigger,使trigger对激活的结果变大'''

    def optimize_backdoor_trigger(self, model: Model, candidate_weights, loader):
        mask, pattern = self.getTrigger()
        pattern.requires_grad = True

        x_top, y_top = 3, 23
        x_bot, y_bot = 6, 26

        cbots, ctops = list(), list()
        for h in range(pattern.size()[0]):
            cbot = (0 - self.mean[h]) / self.std[h]
            ctop = (1 - self.mean[h]) / self.std[h]
            cbots.append(round(cbot, 2))
            ctops.append(round(ctop, 2))

        raw_weights = copy.deepcopy(model.state_dict())
        if isinstance(model, alex_cifar10):
            self.set_handcrafted_filters2(model, candidate_weights, "features.0.weight")
        else:
            self.set_handcrafted_filters2(model, candidate_weights, "conv1.weight")  # 对conv1.weight层进行flip的参数注入

        for epoch in range(2):
            losses = list()
            for i, data in enumerate(loader):
                batch_size = self.conf['batch_size']
                imgs, labels = data
                imgs, labels = imgs.to(self.conf['DEVICE']), labels.to(self.conf['DEVICE'])
                mal_imgs, mal_labels = copy.deepcopy(imgs), copy.deepcopy(labels)
                mal_imgs, mal_labels = mal_imgs.to(self.conf['DEVICE']), mal_labels.to(self.conf['DEVICE'])
                mal_imgs[:batch_size] = (1 - mask) * mal_imgs[:batch_size] + mask * pattern
                mal_labels[:batch_size].fill_(self.conf['MalSetting']['BackdoorLabel'])

                # 对所有层与pattern有关的参数flip后的进行参数注入，加载进模型

                if isinstance(model,alex_cifar10):
                    self.set_handcrafted_filters2(model, candidate_weights, "features.0.weight")
                else:
                    self.set_handcrafted_filters2(model, candidate_weights, "conv1.weight")

                # 使用修改的模型参数计算grad加到trigger中
                # loss, grads = trigger_attention_loss(raw_model, model, backdoor_batch.inputs, pattern, grads=True)
                loss, grads = self.trigger_loss(model, mal_imgs, imgs, pattern, grads=True)
                losses.append(loss.item())
                pattern = pattern + grads[0] * 0.1
                n_channel = pattern.size()[0]

                for h in range(n_channel):
                    pattern[h, x_top:x_bot, y_top:y_bot] = torch.clamp(pattern[h, x_top:x_bot, y_top:y_bot], cbots[h],
                                                                       ctops[h], out=None)

                model.zero_grad()
            self.logger.debug("epoch:{} trigger loss:{}".format(epoch, np.mean(losses)))
        self.logger.debug(pattern[0, x_top:x_bot, y_top:y_bot].cpu().data)
        self.pattern = pattern.clone().detach()
        model.load_state_dict(raw_weights)
        torch.cuda.empty_cache()

    def set_handcrafted_filters2(self, model: Model, candidate_weights, layer_name):
        conv_weights = candidate_weights[layer_name]
        # print("check candidate:",int(torch.sum(conv_weights)))
        model_weights = model.state_dict()
        temp_weights = copy.deepcopy(model_weights[layer_name])
        n_filter = conv_weights.size()[0]

        # 对每层所有参数与pattern方向不同的值进行flip
        # 对反转后的参数取candidate mask位置像素进行还原。
        for i in range(n_filter):
            conv_kernel = model_weights[layer_name][i, ...].clone().detach()
            handcrafted_conv_kernel = self.flip_filter_as_trigger(conv_kernel, difference=None)
            mask = conv_weights[i, ...]
            model_weights[layer_name][i, ...] = mask * handcrafted_conv_kernel + (1 - mask) * model_weights[layer_name][
                i, ...]
        model.load_state_dict(model_weights)
        # n_turn=int(torch.sum(torch.sign(temp_weights)!=torch.sign(model_weights[layer_name])))
        # print("check modify:",n_turn)

    def flip_filter_as_trigger(self, conv_kernel: torch.Tensor, difference):
        flip_factor = 1
        # flip_factor = self.params.flip_factor
        c_min, c_max = conv_kernel.min(), conv_kernel.max()
        if difference is None:
            mask, pattern = self.getTrigger()
            x_top, y_top = 3, 23
            x_bot, y_bot = 6, 26
            pattern = pattern[:, x_top:x_bot, y_top:y_bot]
        else:
            pattern = difference
        w = conv_kernel[0, ...].size()[0]
        resize = transforms.Resize((w, w))
        pattern = resize(pattern)
        p_min, p_max = pattern.min(), pattern.max()
        scaled_pattern = (pattern - p_min) / (p_max - p_min) * (
                c_max - c_min) + c_min  # 对pattern进行归一化的值，用kernel的最小值补上【相对于kernel的trigger变化】

        crop_mask = torch.sign(scaled_pattern) != torch.sign(conv_kernel)  # 得到kernel与scaled_pattern方向不同的坐标
        conv_kernel = torch.sign(scaled_pattern) * torch.abs(conv_kernel)  # 使得kernel与scaled_pattern方向相同
        conv_kernel[crop_mask] = conv_kernel[crop_mask] * flip_factor  # 对与pattern不同向的坐标放大
        return conv_kernel

    def inject_handcrafted_filters(self, model, candidate_weights, loader):
        conv_weight_names = get_conv_weight_names(model)  # 获得conv的名称
        difference = None

        # 每个conv.weigt层
        for layer_name, conv_weights in candidate_weights.items():
            if layer_name not in conv_weight_names:
                continue
            model_weights = model.state_dict()
            n_filter = conv_weights.size()[0]

            # 每个conv.weight层的filter
            # 第一层通过pattern进行
            # 后续层通过上一层的difference差值
            # 通过最大每层良性输入和恶意输入在每层的差值，使得恶意激活参数遇到pattern时的激活值最大。
            for i in range(n_filter):
                conv_kernel = model_weights[layer_name][i, ...].clone().detach()
                handcrafted_conv_kernel = self.flip_filter_as_trigger(conv_kernel, difference)
                # handcrafted_conv_kernel = conv_kernel

                mask = conv_weights[i, ...]
                model_weights[layer_name][i, ...] = mask * handcrafted_conv_kernel + (1 - mask) * \
                                                    model_weights[layer_name][i, ...]
                # model_weights[layer_name][i, ...].mul_(1-mask)
                # model_weights[layer_name][i, ...].add_(mask * handcrafted_conv_kernel)

            model.load_state_dict(model_weights)
            difference = self.conv_activation(model, layer_name, loader, True) - self.conv_activation(model,
                                                                                                      layer_name,
                                                                                                      loader,
                                                                                                      False)

            self.logger.debug("handcraft_conv: {}".format(layer_name))

        torch.cuda.empty_cache()
        if difference is not None:
            feature_difference = self.conv_features(model, loader, True) - self.conv_features(model, loader,
                                                                                              False)
            return feature_difference

    def inject_handcrafted_neurons(self, model, candidate_weights, diff, loader):
        handcrafted_connectvites = defaultdict(list)
        target_label = self.conf['MalSetting']['BackdoorLabel']
        n_labels = -1
        if self.conf['dataset'] == 'cifar10':
            n_labels = 10
        elif self.conf['dataset'] == 'cifar100':
            n_labels = 100

        fc_names = get_neuron_weight_names(model)
        fc_diff = diff  # 特征拉长为一维
        last_layer, last_ids = None, list()

        # 全连接层计算与良性特征和恶意特征 的激活差值相同的符号
        for layer_name, connectives in candidate_weights.items():
            if layer_name not in fc_names:
                continue
            raw_model = copy.deepcopy(model)
            model_weights = model.state_dict()
            ideal_signs = torch.sign(fc_diff)  # 取一维的符号
            # print(ideal_signs)
            n_next_neurons = connectives.size()[0]
            # last_layer
            if n_next_neurons == n_labels:
                break
            # connectives是可用后门参数candidate的掩码
            # 使connectives的符号与良性/恶意特征差值符号相同
            ideal_signs = ideal_signs.repeat(n_next_neurons, 1) * connectives
            # count the flip num
            n_flip = torch.sum(((ideal_signs * torch.sign(model_weights[layer_name]) * connectives == -1).int()))
            self.logger.debug("n_flip in {}:{}".format(layer_name, n_flip))
            model_weights[layer_name] = (1 - connectives) * model_weights[layer_name] + torch.abs(
                connectives * model_weights[layer_name]) * ideal_signs
            model.load_state_dict(model_weights)
            last_layer = layer_name
            fc_diff = self.fc_activation(model, layer_name, loader, attack=True).mean([0]) - self.fc_activation(
                model, layer_name, loader, attack=False).mean([0])

    def conv_activation(self, model, layer_name, loader, attack):
        extractor = FeatureExtractor(model)
        hook = extractor.insert_activation_hook(model)
        module = layer2module(model, layer_name)
        conv_activations = None
        for i, data in enumerate(loader):
            imgs, labels = data
            imgs, labels = imgs.to(self.conf['DEVICE']), labels.to(self.conf['DEVICE'])
            mal_imgs, mal_labels = self.injectTrigger2Imgs(imgs, labels,
                                                           target_label=self.conf['MalSetting']['BackdoorLabel'],
                                                           Test=True)

            _ = model(mal_imgs)
            conv_activation = extractor.activations(model, module)
            conv_activation = torch.mean(conv_activation, [0])
            conv_activations = conv_activation if conv_activations is None else conv_activations + conv_activation

        avg_activation = conv_activations / len(loader)
        extractor.release_hooks()
        torch.cuda.empty_cache()
        return avg_activation

    def fc_activation(self, model: Model, layer_name, loader, attack):
        extractor = FeatureExtractor(model)
        hook = extractor.insert_activation_hook(model)
        module = layer2module(model, layer_name)
        neuron_activations = None
        for i, data in enumerate(loader):
            imgs, labels = data
            imgs, labels = imgs.to(self.conf['DEVICE']), labels.to(self.conf['DEVICE'])
            mal_imgs, mal_labels = self.injectTrigger2Imgs(imgs, labels,
                                                           target_label=self.conf['MalSetting']['BackdoorLabel'],
                                                           Test=True)

            _ = model(mal_imgs)
            neuron_activation = extractor.activations(model, module)
            neuron_activations = neuron_activation if neuron_activations is None else neuron_activations + neuron_activation

        avg_activation = neuron_activations / len(loader)
        extractor.release_hooks()
        torch.cuda.empty_cache()
        return avg_activation

    def conv_features(self, model, loader, attack):
        features = None
        if isinstance(model, CNNMnist):
            for i, data in enumerate(loader):
                imgs, labels = data
                imgs, labels = imgs.to(self.conf['DEVICE']), labels.to(self.conf['DEVICE'])
                mal_imgs, mal_labels = self.injectTrigger2Imgs(imgs, labels,
                                                               target_label=self.conf['MalSetting']['BackdoorLabel'],
                                                               Test=True)
                feature = model.features(mal_imgs).mean([0])
                features = feature if features is None else features + feature
            avg_features = features / len(loader)
        elif isinstance(model, SimpleNet):
            for i, data in enumerate(loader):
                imgs, labels = data
                imgs, labels = imgs.to(self.conf['DEVICE']), labels.to(self.conf['DEVICE'])
                mal_imgs, mal_labels = self.injectTrigger2Imgs(imgs, labels,
                                                               target_label=self.conf['MalSetting']['BackdoorLabel'],
                                                               Test=True)
                feature = model.features(mal_imgs).mean([0])
                features = feature if features is None else features + feature
            avg_features = features / len(loader)
        if isinstance(model, ResNet):
            for i, data in enumerate(loader):
                imgs, labels = data
                imgs, labels = imgs.to(self.conf['DEVICE']), labels.to(self.conf['DEVICE'])
                mal_imgs, mal_labels = self.injectTrigger2Imgs(imgs, labels,
                                                               target_label=self.conf['MalSetting']['BackdoorLabel'],
                                                               Test=True)
                feature = model.features(mal_imgs).mean([0])
                features = feature if features is None else features + feature
            avg_features = features / len(loader)

        return avg_features


if __name__ == '__main__':
    with open(f'../configs/conf.yaml', "r+") as file:
        conf = yaml.load(file, Loader=yaml.FullLoader)
    configure_logger(conf)
    conf['DEVICE'] = 'mps'
    conf['ImageShape'] = (3, 32, 32)
    att = AttF3BA(conf, -1)
    att.initTrigger()
