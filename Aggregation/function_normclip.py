import collections
import copy

import numpy as np
import torch



# @staticmethod
def get_weight_difference(weight1, weight2):
    difference = {}
    res = []
    if type(weight2) == dict or type(weight2) == collections.OrderedDict:
        for name, layer in weight1.items():
            difference[name] = layer.data - weight2[name].data
            res.append(difference[name].view(-1))
    else:
        for name, layer in weight2:
            difference[name] = weight1[name].data - layer.data
            res.append(difference[name].view(-1))

    difference_flat = torch.cat(res)
    return difference, difference_flat

    # @staticmethod


def clip_grad(norm_bound, weight_difference, difference_flat):
    l2_norm = torch.norm(difference_flat.clone().detach().cuda())
    scale = max(1.0, float(torch.abs(l2_norm / norm_bound)))    # 最小值为1.0
    for name in weight_difference.keys():
        if 'weight' in name or 'bias' in name:
            weight_difference[name].div_(scale)
    return weight_difference, l2_norm


def get_l2_norm(weight1, weight2):
    difference = {}
    res = []
    if type(weight2) == dict:
        for name, layer in weight1.items():
            difference[name] = layer.data - weight2[name].data
            res.append(difference[name].view(-1))
    else:
        for name, layer in weight2:
            difference[name] = weight1[name].data - layer.data
            res.append(difference[name].view(-1))

    difference_flat = torch.cat(res)
    l2_norm = torch.norm(difference_flat.clone().detach().cuda())
    l2_norm_np = np.linalg.norm(difference_flat.cpu().numpy())
    return l2_norm, l2_norm_np


class Normclip():
    @classmethod
    def run(cls, cfg, global_model_copy, weights, ID):
        weight_difference, difference_flat = get_weight_difference(global_model_copy,
                                                                   weights[ID])
        clipped_weight_difference, _ = clip_grad(cfg.normclip_ratio, weight_difference, difference_flat)
        weight_difference, difference_flat = get_weight_difference(global_model_copy,
                                                                   clipped_weight_difference)
        weights[ID] = weight_difference

    @classmethod
    def get_l2_norm(cls,  cfg, global_model_copy, weights, ID):
        l2_norm, l2_norm_np = get_l2_norm(global_model_copy,
                                          weights[ID])
        # if ID not in cfg.mal_id:
        #     l2_norm = l2_norm.item()
        # else:
        #     l2_norm = 0
        l2_norm = l2_norm.item()
        return l2_norm