from operator import add

import numpy as np
import torch
import torchvision
from torch import optim, nn
from tqdm import tqdm

from Terminal.malclient import MalClient
from utils.function_normalization import tensor2Normalize


class MalClientFlinvert(MalClient):
    def __init__(self,  cfg, ID, train_dataset, test_dataset, identity="None"):
        super(MalClientFlinvert, self).__init__( cfg, ID, train_dataset, test_dataset, identity="flinvert")

        # self.pattern = torch.rand(self.train_dataset[0][0].shape).to(cfg.device)
        self.pattern = torch.randn(self.train_dataset[0][0].shape).to(cfg.device)
        epsilon = float(eval(cfg.epsilon))
        cfg.eps = epsilon
        self.pattern = (self.pattern) * cfg.eps
        self.mask = torch.ones((cfg.image_shape[-2], cfg.image_shape[-1])).to(cfg.device)

        # print(f"init trigger is \n {(self.pattern)[0, 3:7, 5:9]}")

        if cfg.normalize:
            self.pattern = tensor2Normalize(self.pattern, DEVICE=cfg.device, dataset=cfg.dataset)

    def malicious_train(self, cfg):

        for e in range(cfg.local_epoch_mal):
            trigger = [self.pattern, self.mask]
            trigger = self.generate_trigger(cfg, self.local_model, self.dataloader, poison_method=self.triggerInjectionflinvert, trigger = trigger)

            self.pattern, self.mask = trigger[0], trigger[1]
            trigger = [self.pattern, self.mask]
            self.train(cfg, self.local_model, self.dataloader, poison_method=self.triggerInjectionflinvert,trigger=trigger)
        return self.local_model

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
        for e in range(1):
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
                    poison_images, poison_targets = poison_method(cfg, batch, trigger, IsTest=False)
                    data, target = poison_images.to(torch.float32).to(device), poison_targets.to(device)

                optimizer.zero_grad()
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

        # print(f"|---Trigger Mal performance:")
        # print(f"|------Train acc is : {correct / datasize:.4f}({correct}/{datasize})")
        # print(f"|------Train asr is : {correct_asr / datasize:.4f}({correct_asr}/{datasize})")

        # tq.set_description(f'Epoch [{epoch}/{num_epochs}]')
        # tq.set_postfix(loss=loss.item(), acc=running_train_acc)

    def generate_trigger(self, cfg, model, dataloader, poison_method=None,trigger=None):

        epoch = cfg.epoch_trigger
        device = cfg.device
        epsilon = cfg.eps
        # pattern = trigger[0].to(device)
        pattern = trigger[0]
        # pattern = torch.clamp(pattern, -epsilon, epsilon)
        pattern = torch.nn.Parameter(pattern.to(device))
        pattern.requires_grad = True
        mask = trigger[1].to(device)
        optimizer_trigger = optim.Adam([pattern], lr=cfg.lr_trigger)
        criterion = torch.nn.CrossEntropyLoss()

        model.eval()
        model.to(device)
        for e in range(epoch):
            correct, correct_asr, total_loss, total_loss_asr = 0, 0, 0.0, 0.0
            datasize = len(dataloader.dataset)
            for batch_idx, batch in enumerate(dataloader):
                data, target = batch[0].to(device), batch[1].to(device)
                poison_images, poison_targets = poison_method(cfg, batch, trigger, IsTest=True)
                poison_images, poison_targets = poison_images.to(torch.float32).to(device), poison_targets.to(device)

                optimizer_trigger.zero_grad()
                output = model(poison_images)
                loss1 = criterion(output, poison_targets)
                loss2 = torch.nn.functional.l1_loss(poison_images, data)
                # loss2 = F.mse_loss(poison_images, data)  # 使用均方误差（L2 损失）

                # loss3 = F.l
                loss = loss1 + loss2
                loss.backward()
                optimizer_trigger.step()
                # 优化步骤后裁剪 pattern
                with torch.no_grad():
                    pattern.clamp_(-epsilon, epsilon)
                    # pattern = torch.clamp(pattern, -epsilon, epsilon)
                trigger = [pattern, mask]
                # print(pattern)
                ###############
                pred = output.data.max(1)[1]
                # correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
                correct_asr += pred.eq(poison_targets.data.view_as(pred)).cpu().sum().item()
                total_loss_asr += torch.nn.functional.cross_entropy(output, poison_targets, reduction='sum').item()

                with torch.no_grad():
                    output = model(data)
                    pred = output.data.max(1)[1]
                    correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
                    total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
        # poison_images, poison_targets = poison_method(cfg, batch, trigger, MalID=ID, IsTest=True)
        # poison_images, poison_targets = poison_images.to(torch.float32).to(device), poison_targets.to(device)
        # visual1 = data[:3]
        # visual2 = poison_images[:3]
        # visual3 = visual1 - visual2

        # save_image = torch.cat([visual1, visual2, visual3, visual3 * 5, visual3 * 30], dim=0)
        # save_image = tensor2Denormalize(save_image, DEVICE=cfg.device, dataset=cfg.dataset)
        # torchvision.utils.save_image(save_image, "visual/trigger.png", nrow=3)
        # print(f"|---Generate Trigger")
        # print(f"|---Train acc is : {correct / datasize:.4f}({correct}/{datasize})")
        # print(f"|---Train asr is : {correct_asr / datasize:.4f}({correct_asr}/{datasize})")
        pattern = pattern.detach()
        trigger = [pattern, mask]
        return trigger

    def triggerInjectionflinvert(self, cfg, batch, trigger, IsTest=False):
        pattern = trigger[0].to(cfg.device)
        mask = trigger[1].to(cfg.device)
        imgs, labels = batch[0].to(cfg.device), batch[1].to(cfg.device)


        poison_num = int(imgs.shape[0] * cfg.poison_ratio) if IsTest is False else int(imgs.shape[0])

        imgs[:poison_num] = imgs[:poison_num] + pattern * mask
        labels[:poison_num] = cfg.target_label
        imgs = torch.clamp(imgs, cfg.lower, cfg.upper)
        return imgs, labels


    def backdoor_inject(self, backdoor_params, delta=0.1):
        if self.cfg.model == "vgg11":
            layers = ['features.0', 'features.4', 'features.8', 'features.11', 'features.15', 'features.18',
                      'features.22', 'features.25', 'classifier.0', 'classifier.2', 'classifier.4']
        else:
            layers = ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17',
                      'features.20', 'features.23', 'features.27', 'features.30', 'features.33', 'features.36',
                      'features.40', 'features.43', 'features.46', 'features.49', 'classifier.0', 'classifier.2',
                      'classifier.4']

        model = self.local_model
        trigger = self.pattern
        images = [self.train_dataset[i][0] for i in range(len(self.train_dataset))]

        if self.cfg.model == 'resnet18':
            orig_params = []
            for p in self.local_model.parameters():
                if len(p.shape) != 1:
                    orig_params.append(p)
        elif self.cfg.model == 'vgg11':
            orig_params = []
            for f in model.features:
                if isinstance(f, nn.Conv2d):
                    orig_params.append(f.weight)
                    orig_params.append(f.bias)
            for c in model.classifier:
                if isinstance(c, nn.Linear):
                    orig_params.append(c.weight)
                    orig_params.append(c.bias)
        else:
            raise ValueError('Model not supported')

        for _ in range(10):
            backdoor_modification_sign = self.get_backdoor_modification_sign(model, images, trigger)
            params = []
            if self.cfg.model == 'resnet18':
                for p in model.parameters():
                    if len(p.shape) != 1:
                        params.append(p)
            elif self.cfg.model == 'vgg11':
                params = []
                for f in model.features:
                    if isinstance(f, nn.Conv2d):
                        params.append(f.weight)
                        params.append(f.bias)
                for c in model.classifier:
                    if isinstance(c, nn.Linear):
                        params.append(c.weight)
                        params.append(c.bias)
            else:
                raise ValueError('Model not supported')
            backdoor_done = []
            for i in range(len(params)):
                change = backdoor_modification_sign[i] * delta
                backdoor_mod = backdoor_params[i].astype(int)
                backdoored = np.clip(params[i].cpu().detach().numpy() - backdoor_mod * change,
                                     orig_params[i].cpu().detach().numpy() - delta,
                                     orig_params[i].cpu().detach().numpy() + delta)
                backdoor_done.append(backdoored)
            model_dict = model.state_dict()
            model_keys = model.state_dict().keys()
            count = 0
            for item in model_keys:
                if self.cfg.model == 'cnn':
                    if ('conv' in item or 'shortcut.0' in item or 'linear.weight' in item) and 'weight' in item:
                        model_dict[item] = torch.tensor(backdoor_done[count])
                        count += 1
                elif self.cfg.model == 'resnet18':
                    if 'conv' in item or 'shortcut.0' in item or 'linear.weight' in item:
                        model_dict[item] = torch.tensor(backdoor_done[count])
                        count += 1
                elif self.cfg.model == 'vgg11':
                    for i in range(len(layers)):
                        model_dict[layers[i] + '.weight'] = torch.tensor(backdoor_done[2 * i])
                        model_dict[layers[i] + '.bias'] = torch.tensor(backdoor_done[2 * i + 1])

            model.load_state_dict(model_dict)

    def get_backdoor_modification_sign(self, model, images, trigger):
        grads = None
        num_images = 100 if len(images) > 100 else len(images)
        for index in range(num_images):
            inputs = (torch.tensor(images[index][None, :]).to(self.cfg.device) + torch.tensor(trigger)).to(
                self.cfg.device)
            outputs = model(inputs)
            g = []
            loss = torch.nn.functional.cross_entropy(outputs, torch.tensor([self.cfg.target_label]).to(self.cfg.device), reduction='sum')
            loss.backward()
            if self.cfg.model == 'resnet18':
                for param in model.parameters():
                    if len(param.shape) != 1:
                        g.append(param.grad)
            elif self.cfg.model == 'vgg11':
                for f in model.features:
                    if isinstance(f, nn.Conv2d):
                        g.append(f.weight.grad)
                        g.append(f.bias.grad)
                for c in model.classifier:
                    if isinstance(c, nn.Linear):
                        g.append(c.weight.grad)
                        g.append(c.bias.grad)
            else:
                raise ValueError('Model not supported')
            if grads is None:
                grads = g
            else:
                grads = list(map(add, grads, g))
        grads[:] = [(x / len(grads)).cpu().numpy() for x in grads]
        return grads
