import os
import sys
import torch
from typing import Any
from os.path import dirname as up

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


def get_metrics_name_for_adv(resnet_metrics: Metrics, adv_metrics: Metrics) -> str:
    """ get all metrics name """
    add_name = lambda model_name, metric_name: f'{model_name}_{metric_name}'
    add_resnet = lambda metric_name: add_name(model_name='resnet', metric_name=metric_name)
    add_adv = lambda metric_name: add_name(model_name='adv', metric_name=metric_name)
    metrics_name = list(map(add_resnet, resnet_metrics.get_names())) + \
                   list(map(add_adv, adv_metrics.get_names()))
    return metrics_name