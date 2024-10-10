import copy
from collections import OrderedDict, defaultdict
from typing import Callable

import numpy as np
import torch
import torchvision
from sympy.codegen import Print
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from Models.Alex import alex_cifar10
from Models.extractor import simplenetactivations, resnet18activations, resnetactivations
from Models.init_model import CNNMnist
from Models.model import Model
from Models.resnet_cifar import ResNet, layer2module
from Models.simple import SimpleNet
from Terminal.malclient import MalClient
from utils.function_normalization import tensor2Denormalize
from Functions.data import Subset

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
class MalClientF3BA(MalClient):
    def __init__(self, cfg, ID, train_dataset, test_dataset, identity="None"):
        super(MalClientF3BA, self).__init__(cfg, ID, train_dataset, test_dataset, identity=identity)
        image_shape = train_dataset[0][0].shape

        coordinate_f3ba = [
            [3, 6], [3, 7], [3, 8],
            [4, 6], [4, 7], [4, 8],
            [5, 6], [5, 7], [5, 8],
        ]

        self.mask = torch.zeros_like(train_dataset[0][0])[0]
        coordinate = coordinate_f3ba
        for idx, (i, j) in enumerate(coordinate):
            self.mask[i, j] = 1
        self.mask = self.mask.to(cfg.device)

        torch.manual_seed(111)
        # pattern_tensor = torch.rand(trigger_size)
        self.pattern = torch.rand(self.train_dataset[0][0].shape).to(cfg.device)

        self.pattern = (self.pattern * 255).floor() / 255

        self.pattern = self.mask * self.pattern


        print(f"init trigger is \n {(self.mask * self.pattern)[0, 3:7, 5:9]}")


        self.x_top = 3
        self.x_bot = 6
        self.y_top = 6
        self.y_bot = 9

        if cfg.n_client == 1000:
            nt = int(len(self.train_dataset) * 0.5)
        else:
            nt = int(len(self.train_dataset) * 0.2)

        handcraft_dataset = []
        for i, batch in enumerate(self.train_dataset):
            if i <= nt:
                handcraft_dataset.append([self.train_dataset[i][0], self.train_dataset[i][1]])
        self.handcraft_dataset = Subset(handcraft_dataset)
        self.train_loader = DataLoader(self.train_dataset, batch_size=cfg.batch_size, shuffle=True)
        self.handcraft_loader = DataLoader(self.handcraft_dataset, batch_size=cfg.batch_size, shuffle=True)
        self.handcraft_rnd = 0
        self.previous_global_model = None

    def malicious_train(self, cfg):
        previous_global_model = copy.deepcopy(self.local_model)
        self.handcraft(self.local_model, self.handcraft_loader)
        trigger = [self.pattern, self.mask]
        self.train(cfg, self.local_model, self.dataloader, poison_method=self.triggerInjection, trigger=trigger)

    def train(self, cfg, model, dataloader, optimizer=None, *args, **kwargs):
        poison_method = kwargs.get("poison_method", None)
        trigger = kwargs.get("trigger", None)

        device = cfg.device
        epoch = cfg.local_epoch_mal
        lr = cfg.lr_poison
        weight_decay = cfg.decay_poison
        momentum = cfg.momentum_poison
        if optimizer is None:
            optimizer = optim.SGD(model.parameters(), lr=lr,
                                  weight_decay=weight_decay,
                                  momentum=momentum)

        criterion = torch.nn.CrossEntropyLoss()
        model.train()
        model.to(device)
        for e in range(epoch):
            correct, correct_asr, total_loss, total_loss_asr = 0, 0, 0.0, 0.0
            datasize = len(dataloader.dataset)
            # for batch_idx, batch in enumerate(dataloader):
            tq = tqdm(dataloader, desc=f"Epoch {self.current_epoch} Train", disable=True)
            for batch_idx, batch in enumerate(tq):
                tq.update(1)
                data, target = batch[0].to(device), batch[1].to(device)
                data_ori = data
                target_ori = target
                if data.shape[0] == 1:
                    continue

                if poison_method is not None:
                    # print((trigger[0]*trigger[1])[:,3:6,6:9])
                    poison_images, poison_targets = poison_method(cfg, batch, trigger, IsTest=False)
                    data, target = poison_images.to(torch.float32).to(device), poison_targets.to(device)

                # if batch_idx == 0:
                #     print(f"|---  \n {data[0, 0, 3:7, 5:16]}")
                optimizer.zero_grad()

                # if batch_id == 0:
                #     print(f"|---  \n {data[0, 0, 3:16, 5:16]}")
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                ##########

                pred = output.data.max(1)[1]
                correct += pred.eq(target_ori.data.view_as(pred)).cpu().sum().item()
                total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
                with torch.no_grad():
                    output = model(data_ori)
                    pred = output.data.max(1)[1]
                    correct_asr += pred.eq(target.data.view_as(pred)).cpu().sum().item()
                    total_loss_asr += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()

                if batch_idx == 0:
                    if cfg.normalize:
                        data_ori[:7] = tensor2Denormalize(data_ori[:7], DEVICE=cfg.device, dataset=cfg.dataset)
                        data[:7] = tensor2Denormalize(data[:7], DEVICE=cfg.device, dataset=cfg.dataset)
                    torchvision.utils.save_image(
                        torch.cat([data_ori[:7], data[:7], data_ori[:7] - data[:7], (data_ori[:7] - data[:7]) * 10,
                                   (data_ori[:7] - data[:7]) * 100], dim=0),
                        f"visual/f3ba/triggers/trigger.png", nrow=7)

    def handcraft(self, model, handcraft_loader, ):
        self.handcraft_rnd = self.handcraft_rnd + 1
        model = self.local_model
        model.eval()
        handcraft_loader, train_loader = self.handcraft_loader, self.train_loader

        if self.previous_global_model is None:
            self.previous_global_model = copy.deepcopy(model)
            return
        candidate_weights = self.search_candidate_weights(model, proportion=0.1)
        self.previous_global_model = copy.deepcopy(model)

        print("Optimize Trigger:")
        self.optimize_backdoor_trigger(model, candidate_weights, handcraft_loader)

        print("Inject Candidate Filters:")
        diff = self.inject_handcrafted_filters(model, candidate_weights, handcraft_loader)

        if diff is not None and self.handcraft_rnd % 3 == 1:
            print("Rnd {}: Inject Backdoor FC".format(self.handcraft_rnd))
            self.inject_handcrafted_neurons(model, candidate_weights, diff, handcraft_loader)

    def search_candidate_weights(self, model: Model, proportion=0.2):
        # assert self.kernel_selection in ['random', 'movement']
        candidate_weights = OrderedDict()
        model_weights = model.state_dict()

        n_labels = 0

        if True:  # self.kernel_selection == "movement":
            history_weights = self.previous_global_model.state_dict()
            for layer in history_weights.keys():
                if 'conv' in layer:
                    proportion = 0.02
                elif 'fc' in layer:
                    proportion = 0.001
                model_weights[layer] = model_weights[layer].to(self.cfg.device)
                history_weights[layer] = history_weights[layer].to(self.cfg.device)

                candidate_weights[layer] = (model_weights[layer] - history_weights[layer]) * model_weights[layer]
                n_weight = candidate_weights[layer].numel()
                theta = torch.sort(candidate_weights[layer].flatten(), descending=False)[0][int(n_weight * proportion)]
                candidate_weights[layer][candidate_weights[layer] < theta] = 1
                candidate_weights[layer][candidate_weights[layer] != 1] = 0

        return candidate_weights

    def optimize_backdoor_trigger(self, model: Model, candidate_weights, loader):
        pattern, mask = self.pattern, self.mask
        pattern.requires_grad = True

        x_top, y_top = self.x_top, self.y_top
        x_bot, y_bot = self.x_bot, self.y_bot

        cbots, ctops = [0, 0, 0], [1, 1, 1]
        # cbots, ctops = list(), list()
        # for h in range(pattern.size()[0]):
        #     cbot = (0 - task.means[h]) / task.lvars[h]
        #     ctop = (1 - task.means[h]) / task.lvars[h]
        #     cbots.append(round(cbot, 2))
        #     ctops.append(round(ctop, 2))

        raw_weights = copy.deepcopy(model.state_dict())
        # self.set_handcrafted_filters2(model, candidate_weights, "conv1.weight")
        if isinstance(model, ResNet):
            self.set_handcrafted_filters2(model, candidate_weights, "conv1.weight")  # 对conv1.weight层进行flip的参数注入
        else:
            self.set_handcrafted_filters2(model, candidate_weights, "features.0.weight")


        for epoch in range(2):
            losses = list()
            total_image_num = 0

            for i, data in enumerate(loader):
                batch_size = self.cfg.batch_size
                total_image_num += batch_size

                # clean_batch, backdoor_batch = task.get_batch(i, data), task.get_batch(i, data)
                #
                # backdoor_batch.inputs[:batch_size] = (1 - mask) * backdoor_batch.inputs[:batch_size] + mask * pattern
                # backdoor_batch.labels[:batch_size].fill_(self.cfg.target_label)

                imgs, labels = data
                imgs, labels = imgs.to(self.cfg.device), labels.to(self.cfg.device)
                mal_imgs, mal_labels = copy.deepcopy(imgs), copy.deepcopy(labels)
                mal_imgs, mal_labels = mal_imgs.to(self.cfg.device), mal_labels.to(self.cfg.device)
                mal_imgs[:batch_size] = (1 - mask) * mal_imgs[:batch_size] + mask * pattern

                mal_labels[:batch_size].fill_(self.cfg.target_label)

                # self.set_handcrafted_filters2(model, candidate_weights, "conv1.weight")
                if isinstance(model, ResNet):
                    self.set_handcrafted_filters2(model, candidate_weights, "conv1.weight")  # 对conv1.weight层进行flip的参数注入
                else:
                    self.set_handcrafted_filters2(model, candidate_weights, "features.0.weight")

                if i == 0:
                    torchvision.utils.save_image(mal_imgs,"./visual/f3ba/triggers/mal_imgs.png")
                    torchvision.utils.save_image(imgs,"./visual/f3ba/triggers/imgs.png")
                # break

                # loss, grads = trigger_attention_loss(raw_model, model, backdoor_batch.inputs, pattern, grads=True)
                loss, grads = self.trigger_loss(model, mal_imgs, imgs, pattern, grads=True)
                losses.append(loss.item())
                # if i == 1:
                #     print(grads[0][:, x_top:x_bot, y_top:y_bot])
                # print(grads[0][:, x_top:x_bot, y_top:y_bot])
                pattern = pattern + grads[0] * 0.0001

                n_channel = pattern.size()[0]
                for h in range(n_channel):
                    # print(pattern[0, x_top:x_bot, y_top:y_bot].cpu().data)
                    pattern[h, x_top:x_bot, y_top:y_bot] = torch.clamp(pattern[h, x_top:x_bot, y_top:y_bot], cbots[h],
                                                                       ctops[h], out=None)

                model.zero_grad()
            print(f"数据量是：{total_image_num},总量是{len(self.train_dataset)}")

            print("epoch:{} trigger loss:{}".format(epoch, np.mean(losses)))

        print(pattern[0, x_top:x_bot, y_top:y_bot].cpu().data)

        self.pattern = pattern.clone().detach()
        model.load_state_dict(raw_weights)
        torch.cuda.empty_cache()

    def set_handcrafted_filters2(self, model: Model, candidate_weights, layer_name):
        conv_weights = candidate_weights[layer_name]
        # print("check candidate:",int(torch.sum(conv_weights)))
        model_weights = model.state_dict()
        temp_weights = copy.deepcopy(model_weights[layer_name])

        n_filter = conv_weights.size()[0]

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
        flip_factor = flip_factor
        c_min, c_max = conv_kernel.min(), conv_kernel.max()
        pattern = None
        if difference is None:
            pattern_layers, _ = self.pattern, self.mask
            x_top, y_top = self.x_top, self.y_top
            x_bot, y_bot = self.x_bot, self.y_bot
            pattern = pattern_layers[:, x_top:x_bot, y_top:y_bot]
        else:
            pattern = difference
        w = conv_kernel[0, ...].size()[0]
        resize = transforms.Resize((w, w))
        pattern = resize(pattern)
        p_min, p_max = pattern.min(), pattern.max()
        scaled_pattern = (pattern - p_min) / (p_max - p_min) * (c_max - c_min) + c_min

        crop_mask = torch.sign(scaled_pattern) != torch.sign(conv_kernel)
        conv_kernel = torch.sign(scaled_pattern) * torch.abs(conv_kernel)
        conv_kernel[crop_mask] = conv_kernel[crop_mask] * flip_factor
        return conv_kernel

    def inject_handcrafted_filters(self, model, candidate_weights, loader):
        conv_weight_names = get_conv_weight_names(model)
        difference = None
        for layer_name, conv_weights in candidate_weights.items():
            if layer_name not in conv_weight_names:
                continue
            model_weights = model.state_dict()
            n_filter = conv_weights.size()[0]
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

            print("handcraft_conv: {}".format(layer_name))

        torch.cuda.empty_cache()
        if difference is not None:
            feature_difference = self.conv_features(model, loader, True) - self.conv_features(model, loader,
                                                                                                    False)
            return feature_difference

    def conv_features(self, model, loader, attack):
        features = None
        if isinstance(model, CNNMnist):
            for i, data in enumerate(loader):
                imgs, labels = data
                imgs, labels = imgs.to(self.cfg.device), labels.to(self.cfg.device)
                mal_imgs, mal_labels = self.triggerInjection(self.cfg, (imgs, labels),
                                                             [self.pattern, self.mask],
                                                             IsTest=True)
                feature = model.features(mal_imgs).mean([0])
                features = feature if features is None else features + feature
            avg_features = features / len(loader)
        elif isinstance(model, SimpleNet):
            for i, data in enumerate(loader):
                imgs, labels = data
                imgs, labels = imgs.to(self.cfg.device), labels.to(self.cfg.device)
                mal_imgs, mal_labels = self.triggerInjection(self.cfg, (imgs, labels),
                                                             [self.pattern, self.mask],
                                                             IsTest=True)
                feature = model.features(mal_imgs).mean([0])
                features = feature if features is None else features + feature
            avg_features = features / len(loader)
        if isinstance(model, ResNet):
            for i, data in enumerate(loader):
                imgs, labels = data
                imgs, labels = imgs.to(self.cfg.device), labels.to(self.cfg.device)
                mal_imgs, mal_labels = self.triggerInjection(self.cfg, (imgs, labels),
                                                             [self.pattern, self.mask],
                                                             IsTest=True)
                feature = model.features(mal_imgs).mean([0])
                features = feature if features is None else features + feature
            avg_features = features / len(loader)
        return avg_features

    def conv_activation(self, model, layer_name, loader, attack):
        extractor = FeatureExtractor(model)
        hook = extractor.insert_activation_hook(model)
        module = layer2module(model, layer_name)
        conv_activations = None
        for i, data in enumerate(loader):
            imgs, labels = data
            imgs, labels = imgs.to(self.cfg.device), labels.to(self.cfg.device)
            mal_imgs, mal_labels = self.triggerInjection(self.cfg, (imgs, labels),
                                                         [self.pattern, self.mask],
                                                         IsTest=True)

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
            imgs, labels = imgs.to(self.cfg.device), labels.to(self.cfg.device)
            mal_imgs, mal_labels = self.triggerInjection(self.cfg, (imgs, labels),
                                                         [self.pattern, self.mask],
                                                         IsTest=True)

            _ = model(mal_imgs)
            neuron_activation = extractor.activations(model, module)
            neuron_activations = neuron_activation if neuron_activations is None else neuron_activations + neuron_activation

        avg_activation = neuron_activations / len(loader)
        extractor.release_hooks()
        torch.cuda.empty_cache()
        return avg_activation

    def triggerInjection(self, cfg, batch, trigger, IsTest=False):
        pattern = trigger[0].to(cfg.device)
        mask = trigger[1].to(cfg.device)
        imgs, labels = batch[0].to(cfg.device), batch[1].to(cfg.device)

        # index = np.where(labels.cpu() != cfg.target_label)[0]
        # imgs, labels = imgs[index], labels[index]

        poison_num = int(imgs.shape[0] * cfg.poison_ratio) if IsTest is False else int(imgs.shape[0])
        # poison_num = 5  if IsTest is False else int(imgs.shape[0])
        imgs[:poison_num] = imgs[:poison_num] * (1 - mask) + pattern * mask
        labels[:poison_num] = cfg.target_label
        return imgs, labels

    def inject_handcrafted_neurons(self, model, candidate_weights, diff, loader):
        handcrafted_connectvites = defaultdict(list)
        target_label = self.cfg.target_label
        n_labels = self.cfg.classes

        fc_names = get_neuron_weight_names(model)
        fc_diff = diff
        last_layer, last_ids = None, list()
        for layer_name, connectives in candidate_weights.items():
            if layer_name not in fc_names:
                continue
            raw_model = copy.deepcopy(model)
            model_weights = model.state_dict()
            ideal_signs = torch.sign(fc_diff)
            n_next_neurons = connectives.size()[0]
            # last_layer
            if n_next_neurons == n_labels:
                break

            ideal_signs = ideal_signs.repeat(n_next_neurons, 1) * connectives
            # count the flip num
            n_flip = torch.sum(((ideal_signs * torch.sign(model_weights[layer_name]) * connectives == -1).int()))
            print("n_flip in {}:{}".format(layer_name, n_flip))
            model_weights[layer_name] = (1 - connectives) * model_weights[layer_name] + torch.abs(
                connectives * model_weights[layer_name]) * ideal_signs
            model.load_state_dict(model_weights)
            last_layer = layer_name
            fc_diff = self.fc_activation(model, layer_name, loader, attack=True).mean([0]) - self.fc_activation(
                model, layer_name, loader, attack=False).mean([0])

    def trigger_loss(self, model, backdoor_inputs, clean_inputs, pattern, grads=True):
        model.train()
        backdoor_activations = model.first_activations(backdoor_inputs).mean([0, 1])
        clean_activations = model.first_activations(clean_inputs).mean([0, 1])
        difference = backdoor_activations - clean_activations
        loss = torch.sum(difference * difference)

        # print(f"计算得到的loss是{loss}")
        # print(f"pattern is {pattern[:,3:6,6:9]}")
        if grads:
            grads = torch.autograd.grad(loss, pattern, retain_graph=True)

        # print(f"grad is {grads[0][:,3:6,6:9]}")
        return loss, grads



class FeatureExtractor:
    def __init__(self, model):
        self.layer_names = list()
        layers = dict([*model.named_modules()]).keys()
        # print(layers)
        if isinstance(model, ResNet):
            for layer in layers:

                if 'relu' in layer or layer in resnet18activations or layer in resnetactivations:
                    self.layer_names.append(layer)
        elif isinstance(model, SimpleNet):
            for layer in layers:
                if layer in simplenetactivations:
                    self.layer_names.append(layer)
        elif isinstance(model, CNNMnist):
            for layer in layers:
                if layer in simplenetactivations:
                    self.layer_names.append(layer)
        # elif isinstance(model, alex_cifar10):
        #     for layer in layers:
        #         if layer in simplenetactivations:
        #             self.layer_names.append(layer)

        else:
            raise NotImplemented



        # The activations after convolutional kernel
        self._extracted_activations = dict()
        self._extracted_grads = dict()
        self.hooks = list()

    def clear_activations(self):
        self._extracted_activations = dict()

    def save_activation(self, layer_name: str) -> Callable:
        def hook(module, input, output):
            if layer_name not in self._extracted_activations.keys():
                self._extracted_activations[layer_name] = output

        return hook

    def save_grads(self, layer_name: str) -> Callable:
        def hook(module, input, output):
            self._extracted_grads[layer_name] = output

        return hook

    def insert_activation_hook(self, model: nn.Module):
        named_modules = dict([*model.named_modules()])

        for name in self.layer_names:
            assert name in named_modules.keys()

            layer = named_modules[name]
            hook = layer.register_forward_hook(self.save_activation(name))
            self.hooks.append(hook)

    def insert_grads_hook(self, model: nn.Module):
        named_modules = dict([*model.named_modules()])
        for name in self.layer_names:
            assert name in named_modules.keys()
            layer = named_modules[name]
            layer.register_backward_hook(self.save_grads(name))

    def release_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def activations(self, model, module):
        if isinstance(model, ResNet):
            return self._extracted_activations[module]
        if isinstance(model, SimpleNet):
            return F.relu(self._extracted_activations[module])
        if isinstance(model, CNNMnist):
            return F.relu(self._extracted_activations[module])
        # if isinstance(model, alex_cifar10):
        #     return F.relu(self._extracted_activations[module])
