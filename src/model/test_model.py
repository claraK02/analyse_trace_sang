import os
import sys
import yaml
from icecream import ic 
from easydict import EasyDict
from os.path import dirname as up

import torch
from torch import nn
from torch.nn import Parameter
sys.path.append(up(up(up(os.path.abspath(__file__)))))

from src.model.resnet import get_resnet


def get_dict(resnet: nn.Module) -> dict[str, Parameter]:
    state_dict: dict[str, Parameter] = {}
    for name, param in resnet.named_parameters():
        state_dict[name] = param
    return state_dict


def compare_dict(dict1: dict[str, Parameter],
                 dict2: dict[str, Parameter]
                 ) -> None:
    for name1, param1 in dict1.items():
        # name2 = f'resnet_begin.{name1}'
        name2 = name1
        if name2 in dict2:
            print(f'{name1}: {is_egal(param1, dict2[name2])}')
        else:
            raise ValueError(f'{name2} not in dict2')

def is_egal(tensor1: Parameter, tensor2: Parameter) -> bool:
    tensor_sum = (tensor1 == tensor2).int().sum()
    output = tensor_sum == tensor1.numel()
    return output.item()


if __name__ == '__main__':
    # trained model with all weight
    logging_path = 'logs/resnet_3'
    config = EasyDict(yaml.safe_load(open(os.path.join(logging_path, 'config.yaml'))))
    all_model = get_resnet(config)
    all_weight = torch.load(os.path.join(logging_path, 'all_checkpoint.pt'))
    all_model.load_state_dict(state_dict=all_weight, strict=True)
    all_model_weigth = all_model.state_dict()

    # trained model with only last weight
    last_model = get_resnet(config)
    ic(last_model.state_dict()['fc1.weight'])
    last_weight = torch.load(os.path.join(logging_path, 'checkpoint.pt'))
    last_model.load_dict_learnable_parameters(state_dict=last_weight, strict=True)
    last_model_weigth = last_model.state_dict()

    compare_dict(last_model_weigth, all_model_weigth)

    all_model.eval()
    last_model.eval()

    x = torch.rand((32, 3, 128, 128))
    all_y = all_model.forward(x)
    last_y = last_model.forward(x)
    
    ic(is_egal(all_y, last_y))

    # ic(last_weight['fc1.weight'], all_weight['fc1.weight'])
    # ic(is_egal(last_weight['fc1.weight'], all_weight['fc1.weight']))

    # ic(last_model_weigth['fc1.weight'], all_model_weigth['fc1.weight'])

