import os
import sys
from easydict import EasyDict
from os.path import dirname as up

import torch
from torch import nn
from torch.nn import Parameter
from torchvision import models
from torchvision.models.resnet import ResNet18_Weights

sys.path.append(up(up(up(os.path.abspath(__file__)))))

from src.model.resnet import get_resnet
from utils import utils


def get_dict(resnet: nn.Module) -> dict[str, Parameter]:
    state_dict: dict[str, Parameter] = {}
    for name, param in resnet.named_parameters():
        state_dict[name] = param
    return state_dict

def compare_dict(dict1: dict[str, Parameter],
                 dict2: dict[str, Parameter]
                 ) -> None:
    for name1, param1 in dict1.items():
        name2 = f'resnet_begin.{name1}'
        # name2 = name1
        if name2 in dict2:
            print(f'{name1}: {is_egal(param1, dict2[name2])}')
        else:
            raise ValueError(f'{name2} not in dict2')

def is_egal(tensor1: Parameter, tensor2: Parameter) -> bool:
    tensor_sum = (tensor1 == tensor2).int().sum()
    output = tensor_sum == tensor1.numel()
    return output


if __name__ == '__main__':
    resnet1 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    resnet_begin1 = nn.Sequential(*(list(resnet1.children())[:-1]))

    dict1=get_dict(resnet_begin1)
    print(dict1.keys())
    # weight = torch.load('logs/resnet_3/checkpoint.pt', map_location=torch.device('cpu'))
    # print(weight.keys())

    # print('compare begin')
    # compare_dict(dict1, weight)


    import yaml
    config_path = 'config/config.yaml'   
    config = EasyDict(yaml.safe_load(open(config_path)))
    model = get_resnet(config)
    weight = utils.load_weights(logging_path='logs/resnet_3')
    model.load_dict_learnable_parameters(state_dict=weight, strict=True, verbose=False)
    model_weigth = model.state_dict()
    print(model_weigth.keys())

    compare_dict(dict1, model_weigth)




    



