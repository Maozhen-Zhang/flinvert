import numpy as np
import torch
import torchvision
from torch import optim
from tqdm import tqdm

from Terminal.malclient import MalClient
from utils.function_normalization import tensor2Normalize, tensor2Denormalize


class MalClientDBA(MalClient):
    def __init__(self, cfg, ID, train_dataset, test_dataset, identity="None"):
        super(MalClientDBA, self).__init__(cfg, ID, train_dataset, test_dataset, identity=identity)

        image_shape = train_dataset[0][0].shape
        self.pattern = torch.ones(image_shape).to(cfg.device)
        self.mask = torch.zeros_like(self.pattern)[0]
        if cfg.dataset == 'tiny-imagenet':
            cfg.coordinate_dba = [
                [[3, 6], [3, 7], [3, 8], [3, 9]], [[3, 12], [3, 13], [3, 14], [3, 15]],
                [[6, 6], [6, 7], [6, 8], [6, 9]], [[6, 12], [6, 13], [6, 14], [6, 15]],
            ]
        coordinate = cfg.coordinate_dba[self.ID % 4]
        for idx, (i, j) in enumerate(coordinate):
            self.mask[i, j] = 1
        if cfg.normalize:
            self.pattern = tensor2Normalize(self.pattern, DEVICE=cfg.device, dataset=cfg.dataset)
        print(f"|--- Client {self.ID} init trigger is \n {(self.mask * self.pattern)[0, 3:7, 5:16]}")

    def malicious_train(self, cfg):
        trigger = [self.pattern, self.mask]
        self.train(cfg, self.local_model, self.dataloader, poison_method=self.triggerInjection, trigger=trigger)
        return self.local_model

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

    # def train(self, cfg, current_epoch, model, dataloader, optimizer=None, ID=None, poison_method=None, trigger=None):
    def train(self, cfg, model, dataloader, optimizer=None, *args, **kwargs):
        poison_method = kwargs.get("poison_method", None)
        trigger = kwargs.get("trigger", None)
        IsBackdoor = True if poison_method is not None else False
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
                imgs, labels = batch[0].to(device), batch[1].to(device)
                if imgs.shape[0] == 1:
                    continue

                if poison_method is not None:
                    index = np.where(labels.cpu() != cfg.target_label)[0]
                    data_ori, target_ori = imgs[index], labels[index]
                    poison_num = int(imgs.shape[0] * cfg.poison_ratio)
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

                if IsBackdoor and batch_idx == 0:
                    if cfg.normalize:
                        data_ori[:7] = tensor2Denormalize(data_ori[:7], DEVICE=cfg.device, dataset=cfg.dataset)
                        data[:7] = tensor2Denormalize(data[:7], DEVICE=cfg.device, dataset=cfg.dataset)
                    torchvision.utils.save_image(
                        torch.cat([data_ori[:7], data[:7], data_ori[:7] - data[:7], (data_ori[:7] - data[:7]) * 10,
                                   (data_ori[:7] - data[:7]) * 100], dim=0),
                        f"visual/dba/triggers/trigger.png", nrow=7)
