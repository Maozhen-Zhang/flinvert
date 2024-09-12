import copy
from operator import add

import numpy as np
import torch

from Functions.helper import Helper


def triggerAggergation(cfg, clients):
    pattern = torch.ones(cfg.image_shape).to(cfg.device)
    mask = torch.zeros(cfg.image_shape).to(cfg.device)
    for i in cfg.mal_id[0:4]:
        # pattern = pattern + clients[i].pattern
        mask = mask + clients[i].mask
    # mask = mask / len(cfg.mal_id[0:4])
    # if cfg.normalize:
    #     pattern = tensor2Normalize(pattern, DEVICE=cfg.device, dataset=cfg.dataset)
    # print(f"|--- mask is : {mask}")
    # print(f"|--- pattern is : {pattern}")
    # print(f"|--- trigger is : {(pattern * mask)[0:3:12, 5:12]}")
    return pattern, mask
def triggerAggergationF3BA(cfg, clients):
    pattern = torch.zeros(cfg.image_shape).to(cfg.device)
    mask = torch.zeros(cfg.image_shape).to(cfg.device)
    for i in cfg.mal_id:
        pattern = pattern + clients[i].pattern
        mask = mask + clients[i].mask
    pattern = pattern / len(cfg.mal_id)
    # mask = np.where(np.array(mask.cpu()) > 0.5, 1, 0)
    mask = torch.where(mask > 0.5, torch.tensor(1, dtype=mask.dtype, device=mask.device),
                       torch.tensor(0, dtype=mask.dtype, device=mask.device))
    return pattern, mask

def triggerInjection(cfg, batch, trigger, IsTest=False):
    pattern = trigger[0].to(cfg.device)
    mask = trigger[1].to(cfg.device)
    imgs, labels = batch[0].to(cfg.device), batch[1].to(cfg.device)

    index = np.where(labels.cpu() != cfg.target_label)[0]
    imgs, labels = imgs[index], labels[index]
    # imgs_ori, labels_ori = imgs[index], labels[index]
    poison_num = int(imgs.shape[0] * cfg.poison_ratio) if IsTest is False else int(imgs.shape[0])
    # poison_num = 5  if IsTest is False else int(imgs.shape[0])
    imgs[:poison_num] = imgs[:poison_num] * (1 - mask) + pattern * mask
    labels[:poison_num] = cfg.target_label
    return imgs, labels


def evaluate_trigger(cfg, model, clients, poison_method=None):
    acc_list, loss_list = [], []
    for ID in cfg.mal_id:
        client = clients[ID]
        acc, loss, correct, datasize = Helper.metric(cfg, model, client.dataloader,
                                                     IsBackdoor=True,
                                                     poison_method=poison_method,
                                                     trigger=[client.pattern, client.mask])
        loss_list.append(loss)
        acc_list.append(acc)
    if len(cfg.mal_id) == 4:
        indices = sorted(range(len(acc_list)), key=lambda i: acc_list[i], reverse=True)[:1]
    elif len(cfg.mal_id) == 10:
        indices = sorted(range(len(acc_list)), key=lambda i: acc_list[i], reverse=True)[:1]
    elif len(cfg.mal_id) == 20:
        indices = sorted(range(len(acc_list)), key=lambda i: acc_list[i], reverse=True)[:1]
    elif len(cfg.mal_id) == 40:
        indices = sorted(range(len(acc_list)), key=lambda i: acc_list[i], reverse=True)[:1]
    elif len(cfg.mal_id) == 100:
        indices = sorted(range(len(acc_list)), key=lambda i: acc_list[i], reverse=True)[:10]
    elif len(cfg.mal_id) >= 200:
        indices = sorted(range(len(acc_list)), key=lambda i: acc_list[i], reverse=True)[:20]
    else:
        raise ValueError("mal_id length is not supported")
    # print(f"|---Trigger Mal agg list is {sorted(indices)}:")
    return indices


def evaluate_trigger_by_mal(cfg, model, clients, poison_method=None):
    acc_list, loss_list = [], []
    for ID in cfg.mal_id:
        acc_ = 0
        client = clients[ID]
        trigger = [client.pattern, client.mask]
        for ID_ in cfg.mal_id:
            client_ = clients[ID_]
            acc, loss, correct, datasize = Helper.metric(cfg, model, client_.dataloader,
                                                         IsBackdoor=True,
                                                         poison_method=poison_method,
                                                         trigger=trigger)
            acc_ += acc
        acc_list.append(acc_)
    if len(cfg.mal_id) == 4:
        indices = sorted(range(len(acc_list)), key=lambda i: acc_list[i], reverse=True)[:1]
    elif len(cfg.mal_id) == 10:
        indices = sorted(range(len(acc_list)), key=lambda i: acc_list[i], reverse=True)[:1]
    elif len(cfg.mal_id) == 20:
        indices = sorted(range(len(acc_list)), key=lambda i: acc_list[i], reverse=True)[:1]
    elif len(cfg.mal_id) == 40:
        indices = sorted(range(len(acc_list)), key=lambda i: acc_list[i], reverse=True)[:1]
    elif len(cfg.mal_id) == 100:
        indices = sorted(range(len(acc_list)), key=lambda i: acc_list[i], reverse=True)[:10]
    elif len(cfg.mal_id) >= 200:
        indices = sorted(range(len(acc_list)), key=lambda i: acc_list[i], reverse=True)[:20]
    else:
        raise ValueError("mal_id length is not supported")
    # print(f"|---Trigger Mal agg list is {sorted(indices)}:")
    return indices


def evaluate_trigger_to_mark(cfg, model, clients,trigger, poison_method=None):
    acc_total = 0
    for ID in cfg.mal_id:
        client = clients[ID]
        acc, loss, correct, datasize = Helper.metric(cfg, model, client.dataloader,
                                                     IsBackdoor=True,
                                                     poison_method=poison_method,
                                                     trigger=trigger)
        acc_total += acc
    print(f"|---Trigger acc is {acc_total}")



def triggerInjectionflinvert(cfg, batch, trigger, IsTest=False):
    pattern = trigger[0].to(cfg.device)
    mask = trigger[1].to(cfg.device)
    imgs, labels = batch[0].to(cfg.device), batch[1].to(cfg.device)

    index = np.where(labels.cpu() != cfg.target_label)[0]
    imgs, labels = imgs[index], labels[index]

    poison_num = int(imgs.shape[0] * cfg.poison_ratio) if IsTest is False else int(imgs.shape[0])

    imgs[:poison_num] = imgs[:poison_num] + pattern * mask
    labels[:poison_num] = cfg.target_label
    imgs = torch.clamp(imgs, cfg.lower, cfg.upper)
    return imgs, labels


def triggerAggergationFlinvert(cfg, clients, agg_list):
    pattern = 0
    mask = 0
    for i in agg_list:
        pattern = pattern + clients[i].pattern
        mask = mask + clients[i].mask

    pattern = pattern / len(agg_list)
    mask = mask / len(agg_list)
    return pattern, mask


### inject params
def get_candidate_params(cfg, global_model, global_weight_records):
    # handle param of global model
    params = extract_weights_resnet(global_model)

    if len(global_weight_records) < 10:
        global_weight_records.append(params)
    else:
        global_weight_records.pop(0)
        global_weight_records.append(params)

    if len(global_weight_records) == 10:
        param_no_change = get_param_no_change(cfg, global_weight_records, cfg.threshold)
    elif len(global_weight_records) > 10:
        print("The length of global_weight_records is not enough")
    else:
        param_no_change = None
    return param_no_change


def extract_weights_resnet(model):
    params = []
    for p in model.parameters():
        if len(p.shape) != 1:
            params.append(p.cpu().detach().numpy())
    params = np.array(params, dtype=object)
    # print(f"parames() len is {len(params)}")
    return params


def get_param_no_change(cfg, global_weight_records, threshold):
    # if cfg.arch == "vgg11":
    #     num_layers = 22
    # elif utils.arch == "vgg19":
    #     num_layers = 38
    # elif utils.arch == "resnet18":
    #     num_layers = 21
    # else:
    #     num_layers = 54

    if cfg.model == "resnet18":
        num_layers = 21
    else:
        raise ValueError("model is not supported")
    params = [[] for _ in range(num_layers)]
    for r in range(len(global_weight_records)):
        # 加载global model
        data = global_weight_records[r]
        for i in range(len(data)):
            params[i].append(data[i])
    param_no_change = []
    for i in range(len(params)):
        data = np.array(params[i])
        std = np.std(data, axis=0)
        no_change = std < threshold
        param_no_change.append(no_change)
    return param_no_change


def get_global_grads(global_grads, model, dataset):
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    test_data = iter(test_loader)
    model.eval()
    for _ in range(len(dataset)):
        g = []
        data, target = next(test_data)
        data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target, reduction='sum')
        loss.backward()
        for param in model.parameters():
            if len(param.shape) != 1:
                g.append(param.grad.abs())
        if global_grads is None:
            global_grads = g
        else:
            global_grads = list(map(add, global_grads, g))
    num = len(dataset)
    return global_grads, num


def get_param_not_important_resnet_from_gradsMean(global_grads, datanums):
    global_grads[:] = [x / datanums for x in global_grads]
    param_not_important = []
    for i in range(len(global_grads)):
        data = global_grads[i].cpu().numpy()
        threshold = np.mean(data)
        not_important = data < threshold
        param_not_important.append(not_important)
    return param_not_important
