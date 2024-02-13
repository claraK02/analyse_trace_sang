import os
import sys
import torch
from typing import Any, List
from os.path import dirname as up

from torch.nn import Parameter

sys.path.append(up(os.path.abspath(__file__)))

from src.metrics import Metrics


def get_device(device_config: str) -> torch.device:
    """ get device: cuda or cpu """
    if torch.cuda.is_available() and device_config == 'cuda':
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def put_on_device(device: torch.device, *args: Any) -> None:
    """ put all on device """
    for arg in args:
        arg = arg.to(device)


def get_metrics_name_for_adv(resnet_metrics: Metrics, adv_metrics: Metrics) -> List[str]:
    """ get all metrics name """
    add_name = lambda model_name, metric_name: f'{model_name}_{metric_name}'
    add_resnet = lambda metric_name: add_name(model_name='resnet', metric_name=metric_name)
    add_adv = lambda metric_name: add_name(model_name='adv', metric_name=metric_name)
    metrics_name = list(map(add_resnet, resnet_metrics.get_names())) + \
                   list(map(add_adv, adv_metrics.get_names()))
    return metrics_name


import torch
import os
from torch.nn.parameter import Parameter
from typing import Dict

def load_weights(logging_path: str,
                 model_name: str='res',
                 device: torch.device=torch.device('cpu'),
                 endfile: str='.pt'
                 ) -> Dict[str, Parameter]:
    """
    Load weights from the specified logging path for a given model.

    Args:
        logging_path (str): The path where the weight files are stored.
        model_name (str, optional): The name of the model. Defaults to 'res'.
        device (torch.device, optional): The device to load the weights onto. Defaults to torch.device('cpu').
        endfile (str, optional): The file extension of the weight files. Defaults to '.pt'.

    Returns:
        dict[str, Parameter]: A dictionary containing the loaded weights.

    """
    weight_files = list(filter(lambda x: x.endswith(endfile), os.listdir(logging_path)))
    
    if len(weight_files) == 1:
        weight = torch.load(os.path.join(logging_path, weight_files[0]), map_location=device)
        return weight
    
    model_name_files = list(filter(lambda x: model_name in x, os.listdir(logging_path)))
    if len(model_name_files) > 1:
        raise FileExistsError(f'Confused by multiple weights for the {model_name} model')
    if len(model_name_files) < 1:
        raise FileNotFoundError(f'No weights was found in {logging_path}')
    weight = torch.load(os.path.join(logging_path, model_name_files[0]), map_location=device)
    return weight
