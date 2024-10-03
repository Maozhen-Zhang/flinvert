import argparse
import json

import yaml


class ConfigMapper(object):
    def __init__(self, args):
        for key in args:
            self.__dict__[key] = args[key]

def str_to_list(arg):
    return list(map(int, arg.split(',')))

def parse_args():
    desc = "Pytorch Adversarial Attack"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--config', '-c', type=str, default='train_aisl')
    # parser.add_argument('--dataset','-d', type=str, default='CIFAR10')
    # parser.add_argument('--model','-m', type=str, default='resnet18')
    parser.add_argument('--attack', '-a', type=str)
    parser.add_argument('--defense', '-d', type=str)

    # normclip_ratio setting
    parser.add_argument('--normclip_ratio', '-nr', type=float)

    # flinvert setting
    parser.add_argument('--inject_params', '-ip', action='store_true')
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--delta', type=float)
    parser.add_argument('--inject_epoch', '-ie', type=str_to_list, help='setting wandb name')


    #### neurotoxin setting
    parser.add_argument('--mask_ratio_neurotoxin', type=float)

    parser.add_argument('--wandb', '-w', action='store_true', help='use wandb or not')
    parser.add_argument('--project', '-p', type=str, help='setting wandb project')
    parser.add_argument('--name', '-n', type=str, help='setting wandb name')
    return parser.parse_args()


def get_configs(args):
    # with open(args.cfg_path, "r") as f:
    #     configs = json.load(f)
    with open(f'./configs/{args.config}.yaml', "r") as f:
        configs = yaml.safe_load(f)

    arg_dict = vars(args)  # 返回对象的属性字典  args.xxx ->  arg_dict['xxx']
    for key in arg_dict:
        if key in configs:
            if arg_dict[key] is not None:
                configs[key] = arg_dict[key]

    configs = ConfigMapper(configs)

    if configs.attack == "noatt":
        configs.poison_epoch = [-1, -1]

    if configs.dataset == "cifar10":
        configs.img_size = 32
        configs.classes = 10
    elif configs.dataset == "imagenet10":
        configs.img_size = 224
        configs.classes = 10
    else:
        raise ValueError("dataset not found")



    return configs
