import os
import sys
from tqdm import tqdm
from easydict import EasyDict
from os.path import dirname as up

import torch
from torch import Tensor

sys.path.append(up(os.path.abspath(__file__)))

from config.utils import test_logger
from src.dataloader.dataloader import create_dataloader
from src.model import finetune_resnet
from src.gradcam import GradCam
from src.metrics.metrics import Metrics
from utils import utils


def test(config: EasyDict,
         logging_path: str,
         run_real_data: bool = False,
         run_saliency_metics: bool = False
         ) -> None:
    """
    Run the test on the model using the given configuration.

    Args:
        config (EasyDict): The configuration object.
        logging_path (str): The path to the logging directory.
        run_real_data (bool, optional): Whether to run the test on real data. Defaults to False.
    """

    device = utils.get_device(device_config=config.learning.device)

    # Get data
    test_generator = create_dataloader(config=config,
                                       mode='test',
                                       run_real_data=run_real_data)
    n_test = len(test_generator)

    # Get model
    model = finetune_resnet.get_finetuneresnet(config)
    utils.resume_training(config=config, model=model)
    weight = utils.load_weights(logging_path, device=device, model_name='res')
    model.load_dict_learnable_parameters(state_dict=weight, strict=True)
    model = model.to(device)
    del weight

    # Loss
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    # Get metrics
    metrics = Metrics(num_classes=config.data.num_classes,
                      run_argmax_on_y_true=False,
                      run_acc_per_class=True)
    metrics.to(device)

    test_loss = 0
    test_range = tqdm(test_generator)

    all_y_true: list[Tensor] = []
    all_y_pred: list[Tensor] = []

    model.eval()

    with torch.no_grad():
        for i, item in enumerate(test_range):
            x: Tensor = item['image'].to(device)
            y_true: Tensor = item['label'].to(device)

            y_pred = model.forward(x)

            loss: Tensor = criterion(y_pred, y_true)

            test_loss += loss.item()

            all_y_true.append(y_true.to(torch.device('cpu')))
            all_y_pred.append(y_pred.to(torch.device('cpu')))

            current_loss = test_loss / (i + 1)
            test_range.set_description(f"TEST -> loss: {current_loss:.4f}")
            test_range.refresh()


    ###################################################################
    # Save Scores in logs                                             #
    ###################################################################
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    test_loss = test_loss / n_test
    y_true: Tensor = torch.concat(all_y_true, dim=0).to(device)
    y_pred: Tensor = torch.concat(all_y_pred, dim=0).to(device)
    test_metrics = metrics.compute(y_pred=y_pred, y_true=y_true)
    print(metrics.get_info(metrics_value=test_metrics))

    if 'real' in config.data.path:
        run_real_data = True
    dst_file: str = 'test_log.txt' if not run_real_data else 'test_real_log.txt'
    
    test_logger(path=logging_path,
                metrics=metrics.get_names(),
                values=test_metrics,
                dst_test_name=dst_file)