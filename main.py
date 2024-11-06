
import warnings


# 忽略所有 UserWarning 类型的警告
warnings.filterwarnings("ignore", category=UserWarning)
import copy

import numpy as np
import wandb
from torch.utils.data import DataLoader

from Aggregation.defense import Defense
from Attacks.attacks import Attacks
from Functions.conf import parse_args, get_configs
from Functions.data import get_dataset, get_data_indicates, print_distribution
from Functions.get_model import init_model
from Functions.helper import Helper
from Functions.mk_folders import mk_folder
from Functions.wandb_set import wandb_setting
from Terminal.client import Client
from Terminal.clientNeurotoxin import MalClientNeurotoXin
from Terminal.malclientA3FL import MalClientA3FL
from Terminal.malclientCerp import MalClientCerp
from Terminal.malclientDBA import MalClientDBA
from Terminal.malclientF3BA import MalClientF3BA
from Terminal.malclientFLinvert import MalClientFlinvert
from Terminal.malclientIBA import MalClientIBA
from Terminal.server import Server
from utils.function_backdoor_injection import triggerAggergation, evaluate_trigger, triggerAggergationFlinvert, \
    get_candidate_params, get_global_grads, get_param_not_important_resnet_from_gradsMean, triggerAggergationF3BA, \
    get_global_grads_vgg


class FLTrain():

    def __init__(self, cfg):
        print(cfg.__dict__.items())
        self.cfg = cfg

        self.train_dataset, self.test_dataset, self.dataset_slices = self.datasetinit()
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=256, shuffle=False)

        model = self.getModel()
        tmp_model = copy.deepcopy(model)
        defense = self.getDefense(cfg)

        self.server = self.getServer(model, tmp_model, defense)
        self.clients = self.getClient(cfg, self.dataset_slices, self.test_dataset)

    def datasetinit(self):
        train_dataset, test_dataset = get_dataset(dataset=cfg.dataset, root_path=cfg.root_path,
                                                  IsNormalize=cfg.normalize)
        dataset_slices = get_data_indicates(cfg, train_dataset, test_dataset)
        print_distribution(cfg, dataset_slices)  # print distribution

        return train_dataset, test_dataset, dataset_slices

    def getModel(self):
        model = init_model(self.cfg)
        if self.cfg.load_checkpoint:
            cfg.start_epoch = cfg.load_epoch + 1
            cfg.end_epoch = cfg.epoch
            Helper.load_checkpoint(self.cfg, model)
        else:
            cfg.start_epoch = 0
            cfg.end_epoch = cfg.epoch
        return model

    def getAttack(self, cfg):
        attack = Attacks(cfg, -1, identity=cfg.attack)
        return attack

    def getDefense(self, cfg):
        return Defense(cfg)

    def getServer(self, model, tmp_model, defense):
        server = Server(self.cfg, model, tmp_model, defense)
        return server

    def getClient(self, cfg, dataset_slices, test_dataset):
        clients = {}
        for ID in range(cfg.n_client):
            Clients = {
                "noatt": Client,
                "dba": MalClientDBA,
                "flinvert": MalClientFlinvert,
                "neurotoxin": MalClientNeurotoXin,
                "a3fl": MalClientA3FL,# too slow
                "iba": MalClientIBA, # sample
                "cerp": MalClientCerp,
                "f3ba": MalClientF3BA,
            }
            if len(dataset_slices[ID]) == 0:
                continue
            identity = cfg.attack if ID in cfg.mal_id else "noatt"
            clients[ID] = Clients[identity](cfg, ID, dataset_slices[ID], test_dataset, identity=identity)
            print(f"|---Client {ID} identity is {identity}")

        cfg.mal_id = [id for id in cfg.mal_id if id in clients.keys()]

        return clients

    def updatelr(self, e):
        print(f"|---Epoch {e} defense is {cfg.defense}")
        print(f"|---Epoch {e} attack  is {cfg.attack}")
        print(f"|---Epoch {e} lr      is {cfg.lr}")
        print(f"|---Epoch {e} lr_inv  is {cfg.lr_trigger}") if cfg.attack == "flinvert" else ""


    def train_model(self):
        models = [init_model(self.cfg) for _ in range(self.cfg.sample)]
        global_weight_records = []
        weight_accumulators = {id: self.server.global_model.state_dict() for id in range(cfg.n_client)}

        for e in range(self.cfg.start_epoch, self.cfg.end_epoch):

            ### init
            backdoor_params = []
            norms = {}
            global_grads = None
            self.updatelr(e)
            choice_id, mal_id = self.server.getChoiceId(self.clients)

            # weight_accumulators = {id: self.server.global_model.state_dict() for id in range(cfg.n_client)}
            for i, ID in enumerate(choice_id):
                weight_accumulators[ID] = self.server.global_model.state_dict()

            ### warm up for flinvert
            if e in range(cfg.poison_epoch[0] - 10, cfg.poison_epoch[1]) and cfg.attack == 'flinvert' and cfg.inject_params:
                param_no_change = get_candidate_params(cfg, self.server.global_model, global_weight_records)
                # print(f"这里的参数数量是{len(param_no_change)}") if param_no_change is not None else ""

            datanums = 0
            for i, ID in enumerate(choice_id):
                if ID in mal_id and cfg.inject_params and e in range(cfg.poison_epoch[0], cfg.poison_epoch[1]) and cfg.attack == 'flinvert' and e in range(cfg.inject_epoch[0], cfg.inject_epoch[1]):


                    if cfg.model == 'resnet18':
                        global_grads, datanum = get_global_grads(global_grads, self.server.global_model,
                                                             self.clients[ID].train_dataset)
                    elif cfg.model == 'vgg11':
                        global_grads, datanum = get_global_grads_vgg(global_grads, self.server.global_model,
                                                             self.clients[ID].train_dataset)
                    else:
                        raise ValueError(f"Model {cfg.model} is not supported")
                    datanums += datanum

            if e in range(cfg.poison_epoch[0], cfg.poison_epoch[1]) and cfg.attack == 'flinvert' and cfg.inject_params and e in range(cfg.inject_epoch[0], cfg.inject_epoch[1]):
                if param_no_change is not None and global_grads is not None:
                    param_not_important_global = get_param_not_important_resnet_from_gradsMean(global_grads,
                                                                                               datanums)
                    for idx in range(len(param_not_important_global)):
                        backdoor_params.append(
                            np.logical_and(param_no_change[idx], param_not_important_global[idx]))
                    backdoor_params = np.array(backdoor_params, dtype=object)

                    for i, ID in enumerate(choice_id):
                        if ID in mal_id and len(backdoor_params) > 0:
                            print(f" params inject")
                            client = self.clients[ID]
                            train_model = Helper.copyParams(models[i], weight_accumulators[ID])
                            client.load_model(train_model)
                            client.backdoor_inject(backdoor_params, cfg.delta)
                            client.uploadWeight(weight_accumulators)

            if cfg.clip == True:
                for i, ID in enumerate(choice_id):
                    self.server.normclip.run(self.cfg, self.server.global_model.state_dict(), weight_accumulators, ID)
                    norm = self.server.normclip.get_l2_norm(cfg, self.server.global_model.state_dict(), weight_accumulators, ID)

            for i, ID in enumerate(choice_id):
                client = self.clients[ID]
                client.updateEpoch(e)

                train_model = Helper.copyParams(models[i], weight_accumulators[ID])
                client.load_model(train_model)
                client.local_train(self.cfg)
                client.uploadWeight(weight_accumulators)
            self.server.aggregate_weight(weight_accumulators, choice_id)

            self.metric(e)
            self.metricbackdoor(e, self.clients, choice_id, mal_id) if cfg.attack != "noatt" else ""
            self.saveinfo(e)
            print(" ")

    def metric(self, e):
        test_dataloder = self.test_dataloader
        model = self.server.global_model
        acc, loss, correct, datasize = Helper.metric(self.cfg, model, test_dataloder)
        print(f"|---Epoch {e}, loss  {loss:.6f}, acc: {acc * 100:.4f}%({correct}/{datasize})")
        wandb.log({"acc": acc, "loss": loss, "Epoch": e}) if cfg.wandb else None


    def metricbackdoor(self, e, clients, choice_id, mal_id):
        test_dataloder = self.test_dataloader
        model = self.server.global_model
        if cfg.attack == "neurotoxin":
            trigger = [self.clients[cfg.mal_id[0]].pattern, self.clients[cfg.mal_id[0]].mask]
            from utils.function_backdoor_injection import triggerInjection
            inject_method = triggerInjection
            self.gobal_pattern, self.global_mask = trigger[0], trigger[1]

        elif cfg.attack == "dba":
            trigger = triggerAggergation(self.cfg, clients)
            from utils.function_backdoor_injection import triggerInjection
            inject_method = triggerInjection
            self.gobal_pattern, self.global_mask = trigger[0], trigger[1]

        elif cfg.attack == "f3ba" or cfg.attack == "f3ba_":
            trigger = triggerAggergationF3BA(self.cfg, clients)
            from utils.function_backdoor_injection import triggerInjection
            inject_method = triggerInjection
            self.gobal_pattern, self.global_mask = trigger[0], trigger[1]
            for id in cfg.mal_id:
                clients[id].pattern = self.gobal_pattern.clone().detach()
                clients[id].mask = self.global_mask.clone().detach()

        elif cfg.attack == "a3fl":
            trigger = [self.clients[cfg.mal_id[0]].pattern, self.clients[cfg.mal_id[0]].mask]
            from utils.function_backdoor_injection import triggerInjection
            inject_method = triggerInjection
            self.gobal_pattern, self.global_mask = trigger[0], trigger[1]
            for id in cfg.mal_id:
                clients[id].pattern = self.gobal_pattern.clone().detach()
                clients[id].mask = self.global_mask.clone().detach()

        elif cfg.attack == "iba":
            metric = clients[cfg.mal_id[0]].metric
            asr, loss_asr, correct_asr, datasize_asr = metric(cfg, model, test_dataloder, IsBackdoor=True)
            print(f"|---Epoch {e}, loss  {loss_asr:.6f}, asr: {asr * 100:.4f}%({correct_asr}/{datasize_asr})")
            wandb.log({"asr": asr, "loss_asr": loss_asr, "Epoch": e}) if cfg.wandb else None
            return

        elif cfg.attack == "cerp":
            trigger = triggerAggergation(self.cfg, clients)
            from utils.function_backdoor_injection import triggerInjection
            inject_method = triggerInjection
            self.gobal_pattern, self.global_mask = trigger[0], trigger[1]

        elif cfg.attack == "flinvert":
            from utils.function_backdoor_injection import triggerInjectionflinvert
            inject_method = triggerInjectionflinvert

            indices = evaluate_trigger(cfg, self.server.global_model, clients, poison_method=triggerInjectionflinvert)
            trigger = triggerAggergationFlinvert(self.cfg, clients, indices)

            ## 下次更新的准备
            self.gobal_pattern, self.global_mask = trigger[0], trigger[1]
            for id in cfg.mal_id:
                clients[id].pattern = self.gobal_pattern
                clients[id].mask = self.global_mask

            if e == cfg.start_epoch:
                cfg.initial_lr = 0.01
                cfg.final_lr = 0.001
                cfg.total_iterations = 200
                # cfg.initial_lr = 0.1
                # cfg.final_lr = 0.01
                # cfg.total_iterations = 200
                cfg.decay_rate = (cfg.final_lr / cfg.initial_lr) ** (1 / cfg.total_iterations)
            if cfg.n_client == 20:
                cfg.lr_trigger = cfg.initial_lr * (cfg.decay_rate ** (e - cfg.start_epoch))
            else:
                cfg.lr_trigger = cfg.initial_lr * (cfg.decay_rate ** (e - cfg.start_epoch))

        else:
            raise ValueError(f"Attack {cfg.attack} is not supported")
        asr, loss_asr, correct_asr, datasize_asr = Helper.metric(self.cfg, model, test_dataloder,
                                                                 IsBackdoor=True,
                                                                 poison_method=inject_method,
                                                                 trigger=trigger)
        print(f"|---Epoch {e}, loss  {loss_asr:.6f}, asr: {asr * 100:.4f}%({correct_asr}/{datasize_asr})")
        wandb.log({"asr": asr, "loss_asr": loss_asr, "Epoch": e}) if cfg.wandb else None

    def saveinfo(self, e):

        save_dict = {
            "global_weight": self.server.global_model.state_dict(),
            "global_pattern": self.gobal_pattern if cfg.attack != "noatt" and cfg.attack != 'iba' else None,
            "global_mask": self.global_mask if cfg.attack != "noatt" and cfg.attack != 'iba' else None,
        }
        if cfg.attack != "noatt" and cfg.attack == 'iba':
            save_dict = {
                "global_weight": self.server.global_model.state_dict(),
                "unet": self.clients[cfg.mal_id[0]].unet.state_dict(),
            }



        if (e+1) % 200 == 0 and cfg.attack == 'noatt' and cfg.defense == 'fedavg':
            Helper.saveinfo(cfg, save_dict, e)
        elif e == cfg.epoch-1:
            Helper.saveinfo(cfg, save_dict, e)

if __name__ == '__main__':
    args = parse_args()
    cfg = get_configs(args)
    mk_folder(cfg)
    wandb_setting(cfg)
    train = FLTrain(cfg)
    train.train_model()
