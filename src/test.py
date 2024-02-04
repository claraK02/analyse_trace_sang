import os
import sys
from tqdm import tqdm
from easydict import EasyDict
from os.path import dirname as up
# from icecream import ic

import torch
from torch import Tensor

sys.path.append(up(os.path.abspath(__file__)))

from config.config import test_logger
from src.dataloader.dataloader import create_dataloader
from metrics import Metrics
from src.model import resnet
from utils import utils


def test(config: EasyDict, logging_path: str) -> None:

    device = utils.get_device(device_config=config.learning.device)

    # Get data
    test_generator = create_dataloader(config=config, mode='test')
    n_test = len(test_generator)

    # Get model
    model = resnet.get_resnet(config)
    model.load_dict_learnable_parameters(state_dict=utils.load_weights(logging_path, device=device),
                                         strict=True)
    model = model.to(device)
    # print(model)

    # Loss
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    # Get metrics
    metrics = Metrics(num_classes=config.data.num_classes, run_argmax_on_y_true=False)
    metrics.to(device)

    test_loss = 0
    test_metrics = metrics.init_metrics()
    test_range = tqdm(test_generator)

    model.eval()

    with torch.no_grad():
        for i, (x, y_true, _) in enumerate(test_range):
            x: Tensor = x.to(device)
            y_true: Tensor = y_true.to(device)
            # ic(x[0], y_true, x.shape, y_true.shape)

            y_pred = model.forward(x)
            # ic(y_pred, y_pred.shape)

            loss = criterion(y_pred, y_true)

            test_loss += loss.item()
            test_metrics += metrics.compute(y_pred, y_true)
            # ic(test_loss)
            # ic(test_metrics)
            # exit()

            current_loss = test_loss / (i + 1)
            test_range.set_description(f"TEST -> loss: {current_loss:.4f}")
            test_range.refresh()
            

    ###################################################################
    # Save Scores in logs                                             #
    ###################################################################
    test_loss = test_loss / n_test
    test_metrics = test_metrics / n_test

    test_logger(path=logging_path,
                metrics=metrics.get_names(),
                values=test_metrics)