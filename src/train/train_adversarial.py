import os
import sys
import time
import numpy as np
from tqdm import tqdm
from itertools import chain
from easydict import EasyDict
from os.path import dirname as up

import torch
from torch import Tensor
from torch.optim import Adam

sys.path.append(up(up(up(os.path.abspath(__file__)))))

from config.utils import train_step_logger, train_logger
from src.dataloader.dataloader import create_dataloader
from src.metrics import Metrics
from src.model import finetune_resnet, adversarial
from utils import utils, plot_learning_curves


def train(config: EasyDict, logspath: str = 'logs') -> None:
    """
    Train the adversarial model.

    Args:
        config (EasyDict): Configuration object containing the model and training parameters.
        logspath (str, optional): The path to the logs directory. Defaults to 'logs'.
    
    Raises:
        ValueError: If the model name is not adversarial.
    """
    if config.model.name != 'adversarial':
        raise ValueError(f"Expected model.name=adversarial but found {config.model.name}.")

    # Get data
    train_generator = create_dataloader(config=config,
                                        mode='train',
                                        run_real_data=False)
    val_generator = create_dataloader(config=config,
                                      mode='val',
                                      run_real_data=False)
    n_train, n_val = len(train_generator), len(val_generator) 
    print(f"Found {n_train} training batches and {n_val} validation batches")

    # Get model
    res_model = finetune_resnet.get_finetuneresnet(config)
    adv_model = adversarial.get_adv(config)

    # Loss
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    alpha: float = config.learning.adv.alpha # weight of the adversarial loss

    # Optimizer and Scheduler
    resnet_optimizer = Adam(res_model.get_learned_parameters(),
                            lr=config.learning.learning_rate)
    adv_optimizer = Adam(chain(adv_model.parameters(),
                               res_model.get_learned_parameters()),
                         lr=config.learning.adv.learning_rate_adversary)

    # Get metrics
    res_metrics = Metrics(num_classes=config.data.num_classes,
                          run_argmax_on_y_true=False)
    adv_metrics = Metrics(num_classes=config.data.background_classes,
                          run_argmax_on_y_true=False)
    metrics_name = ['res loss', 'adv loss'] + utils.get_metrics_name_for_adv(res_metrics, adv_metrics)

    # Get and put on device
    device = utils.get_device(device_config=config.learning.device)
    utils.put_on_device(device, res_model, adv_model, res_metrics, adv_metrics)

    # Save experiment
    save_experiment = config.learning.save_experiment
    print(f'{save_experiment = }')
    if save_experiment:
        logging_path = train_logger(config,
                                    metrics_name=metrics_name,
                                    logspath=logspath)
        best_val_loss = 10e6


    ###############################################################
    # Start Training                                              #
    ###############################################################
    start_time = time.time()

    for epoch in range(1, config.learning.epochs + 1):
        print("epoch: ", epoch)
        train_loss = 0
        train_metrics = np.zeros(len(metrics_name))
        train_range = tqdm(train_generator)

        # Training
        res_model.train()
        adv_model.train()
        for i, item in enumerate(train_range):
            x: Tensor = item['image'].to(device)
            res_true: Tensor = item['label'].to(device)
            adv_true: Tensor = item['background'].to(device)

            inter, res_pred = res_model.forward_and_get_intermediare(x)
            adv_pred = adv_model.forward(x=inter)

            res_loss: Tensor = criterion(res_pred, res_true)
            adv_loss: Tensor = criterion(adv_pred, adv_true)

            # crossloss = res_loss - alpha * adv_loss
            crossloss = res_loss / (alpha * adv_loss)

            adv_loss.backward(retain_graph=True)
            crossloss.backward()

            adv_optimizer.step()
            resnet_optimizer.step()

            resnet_optimizer.zero_grad()
            adv_optimizer.zero_grad()

            train_loss += crossloss.item()
            train_metrics += np.concatenate((np.array([res_loss.item(), adv_loss.item()]),
                                             res_metrics.compute(y_pred=res_pred, y_true=res_true),
                                             adv_metrics.compute(y_pred=adv_pred, y_true=adv_true)))

            current_loss = train_loss / (i + 1)
            train_range.set_description(f"TRAIN -> epoch: {epoch} || loss: {current_loss:e} res: {res_loss.item():.2f} adv: {adv_loss.item():.2f}")
            train_range.refresh()

        ###############################################################
        # Start Validation                                            #
        ###############################################################

        val_loss = 0
        val_metrics = np.zeros(len(metrics_name))
        val_range = tqdm(val_generator)

        res_model.eval()
        adv_model.eval()
        with torch.no_grad():
            
            for i, item in enumerate(val_range):
                x: Tensor = item['image'].to(device)
                res_true: Tensor = item['label'].to(device)
                adv_true: Tensor = item['background'].to(device)

                inter, res_pred = res_model.forward_and_get_intermediare(x)
                adv_pred = adv_model.forward(x=inter)

                res_loss = criterion(res_pred, res_true)
                adv_loss = criterion(adv_pred, adv_true)

                # crossloss = res_loss - alpha * adv_loss
                crossloss = res_loss / (alpha * adv_loss)

                val_loss += crossloss.item()
                val_metrics += np.concatenate((np.array([res_loss.item(), adv_loss.item()]),
                                               res_metrics.compute(y_pred=res_pred, y_true=res_true),
                                               adv_metrics.compute(y_pred=adv_pred, y_true=adv_true)))

                current_loss = val_loss / (i + 1)
                val_range.set_description(f"VAL   -> epoch: {epoch} || loss: {current_loss:.2f} res: {res_loss.item():.2f} adv: {adv_loss.item():.2f}")
                val_range.refresh()
          

        ###################################################################
        # Save Scores in logs                                             #
        ###################################################################
        train_loss = train_loss / n_train
        val_loss = val_loss / n_val
        train_metrics = train_metrics / n_train
        val_metrics = val_metrics / n_val

        if save_experiment:
            train_step_logger(path=logging_path, 
                              epoch=epoch, 
                              train_loss=train_loss, 
                              val_loss=val_loss,
                              train_metrics=train_metrics,
                              val_metrics=val_metrics,
                              log_name='train_log.csv')
            
            if val_loss < best_val_loss:
                print('save model weights')
                torch.save(res_model.get_dict_learned_parameters(),
                           os.path.join(logging_path, 'res_checkpoint.pt'))
                torch.save(adv_model.get_dict_learned_parameters(),
                           os.path.join(logging_path, 'adv_checkpoint.pt'))
                best_val_loss = val_loss

            print(f'{best_val_loss = }')

    stop_time = time.time()
    print(f"training time: {stop_time - start_time}secondes for {config.learning.epochs} epochs")
    
    if save_experiment:
        plot_learning_curves.save_learning_curves(path=logging_path)


if __name__ == '__main__':
    import yaml
    config = EasyDict(yaml.safe_load(open('config/config.yaml')))  # Load config file
    config.model.name = 'adversarial'
    train(config=config)