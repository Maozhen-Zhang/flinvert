import copy
import logging
import statistics

import numpy as np
import torch
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus import PrivacyEngine


from Attacks.att_no_attack import NoAtt
from Functions.log import get_logger


class NoAttLDP(NoAtt):
    def __init__(self, conf):
        super(NoAttLDP, self).__init__(conf)
        self.conf = conf
        self.logger = get_logger(conf['logger']['logger_name'])
        self.epsilon = 6
        self.delta = 10 ** (-5)
        self.sigma = np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        self.frac = conf['sample_client'] / conf['n_client']
        self.lr = self.conf['lr']

    def train(self, epoch, model, trainloader, optimizer, criterion, DEVICE):
        model = copy.deepcopy(model)
        # optimizer = torch.optim.SGD(model.parameters(), lr=self.dlr)
        lr = self.conf['lr']
        weight_decay = self.conf['weight_decay']
        momentum = self.conf['momentum']

        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                              weight_decay=weight_decay,
                              momentum=momentum)
        EPSILON = self.epsilon
        DELTA = self.delta
        MAX_GRAD_NORM = 1.0
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=trainloader,
            epochs=epoch,
            target_epsilon=EPSILON,
            target_delta=DELTA,
            max_grad_norm=MAX_GRAD_NORM,
        )
        raw_model = copy.deepcopy(model)
        model.train()
        model.to(DEVICE)
        epoch_loss = []
        MAX_PHYSICAL_BATCH_SIZE = 128
        for e in range(epoch):
            model.zero_grad()
            batch_loss = []
            with BatchMemoryManager(
                        data_loader=train_loader,
                        max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
                        optimizer=optimizer
                ) as memory_safe_data_loader:
                for batch_idx, (data, target) in enumerate(trainloader):
                    data, target = data.to(DEVICE), target.to(DEVICE)
                    if data.shape[0] == 1:
                        continue
                    model.zero_grad()
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)

                    # 清除之前的计算图
                    model.zero_grad()
                    loss.backward()
                    # 清除之前的计算图
                    # torchdp.clear_backprops(model)

                    optimizer.step()
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss) / len(batch_loss))

            # epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(self.delta)
            epsilon = privacy_engine.get_epsilon(DELTA)
            self.logger.info(
                f"\tTrain Epoch: {epoch} \t"
                f"(ε = {epsilon:.2f}, δ = {DELTA})"
            )
        torch.cuda.empty_cache()

    def compute_local_model_update(self, PreModel, LastModel):
        if len(list(PreModel.parameters())) != len(list(LastModel.parameters())):
            raise AssertionError("Two models have different length in number of clients")
        norm_values = []
        for i in range(len(list(PreModel.parameters()))):
            norm_values.append(torch.norm(list(PreModel.parameters())[i] - list(LastModel.parameters())[i]))
        return statistics.median(norm_values)
