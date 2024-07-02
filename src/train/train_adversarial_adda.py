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
from src.metrics.metrics import Metrics
from src.model.adversarial_adda import get_source_cnn, get_target_cnn, get_discriminator
from utils import utils, plot_learning_curves


def print_tensor_info(prefix, tensor):
    print(f"{prefix} - size: {tensor.size()}, data: {tensor}")


def train(config: EasyDict, logspath: str = 'logs') -> None:
    """
    Train the ADDA model.

    Args:
        config (EasyDict): Configuration object containing the model and training parameters.
        logspath (str, optional): The path to the logs directory. Defaults to 'logs'.

    Raises:
        ValueError: If the model name is not adda.
    """
    if config.model.name != 'adda':
        raise ValueError(f"Expected model.name=adda but found {config.model.name}.")

    # Get data
    source_train_loader = create_dataloader(config=config, mode='train', run_real_data=False)
    target_train_loader = create_dataloader(config=config, mode='train', run_real_data=True)
    source_val_loader = create_dataloader(config=config, mode='val', run_real_data=False)
    target_val_loader = create_dataloader(config=config, mode='val', run_real_data=True)

    n_train = len(source_train_loader)
    n_val = len(source_val_loader)
    m_train = len(target_train_loader)
    m_val = len(target_val_loader)
    print(f"Found {n_train} source training batches and {n_val} source validation batches")
    print(f"Found {m_train} target training batches and {m_val} target validation batches")

    # Get model
    source_model = get_source_cnn(config)
    target_model = get_target_cnn(config)
    discriminator = get_discriminator(config)

    source_extractor = source_model.get_feature_extractor()
    source_classifier = source_model.get_classifier()

    target_extractor = target_model.get_feature_extractor()
    target_classifier = target_model.get_classifier()

    # Loss
    classification_criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    adversarial_criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    # Optimizers
    source_optimizer = Adam(source_model.parameters(), lr=config.learning.learning_rate)
    target_optimizer = Adam(target_model.parameters(), lr=config.learning.learning_rate)
    discriminator_optimizer = Adam(discriminator.parameters(), lr=config.learning.adda.learning_rate_adversary)

    # Metrics
    source_metrics = Metrics(num_classes=config.data.num_classes, run_argmax_on_y_true=False)
    discriminator_metrics = Metrics(num_classes=2, run_argmax_on_y_true=False)  # For binary classification (source vs target)
    metrics_name = ['source loss', 'target loss', 'discriminator loss'] + utils.get_metrics_name_for_adv(source_metrics, discriminator_metrics)

    # Device
    device = utils.get_device(device_config=config.learning.device)
    utils.put_on_device(device, source_extractor, source_classifier, target_extractor, target_classifier, discriminator, source_metrics, discriminator_metrics)

    # Save experiment
    save_experiment = config.learning.save_experiment
    if save_experiment:
        logging_path = train_logger(config, metrics_name=metrics_name, logspath=logspath)
        best_val_loss = 10e6

    ###############################################################
    # Start Training                                              #
    ###############################################################
    start_time = time.time()

    for epoch in range(1, config.learning.epochs + 1):
        train_loss = 0
        train_metrics = np.zeros(len(metrics_name))
        train_range = tqdm(zip(source_train_loader, target_train_loader))

        # Training mode
        source_extractor.train()
        source_classifier.train()
        target_extractor.train()
        target_classifier.train()
        discriminator.train()

        for i, (source_item, target_item) in enumerate(train_range):
            source_x = source_item['image'].to(device)
            source_y = source_item['label'].to(device)
            target_x = target_item['image'].to(device)

            # Source CNN forward, classification loss & Update
            # source_intermediate, source_pred = source_model.forward_and_get_intermediare(source_x)
            # source_class_loss = classification_criterion(source_pred, source_y)

            source_intermediate = source_extractor.forward(source_x)
            source_pred = source_classifier(source_intermediate)
            source_class_loss = classification_criterion(source_pred, source_y)

            source_optimizer.zero_grad()
            source_class_loss.backward()
            source_optimizer.step()

            # Detach intermediate source representation to avoid retaining the graph
            source_intermediate = source_intermediate.detach()

            # Discriminator forward pass and adversarial loss for source data
            domain_pred_s = discriminator(source_intermediate)
            adversarial_loss_s = adversarial_criterion(domain_pred_s, torch.ones(domain_pred_s.size(0), dtype=torch.long, device=device))

            # Target CNN forward pass and adversarial loss for target data
            # target_intermediate, _ = target_model.forward_and_get_intermediare(target_x)
            target_intermediate = target_extractor.forward(target_x)
            domain_pred_t = discriminator(target_intermediate)
            adversarial_loss_t = adversarial_criterion(domain_pred_t, torch.zeros(domain_pred_t.size(0), dtype=torch.long, device=device))

            # Update Discriminator
            discriminator_optimizer.zero_grad()
            (adversarial_loss_s + adversarial_loss_t).backward()
            discriminator_optimizer.step()

            # Detach intermediate target representation to avoid retaining the graph (debug)
            target_intermediate = target_intermediate.detach()

            # Target CNN adversarial loss & Update
            # target_intermediate, _ = target_model.forward_and_get_intermediare(target_x)
            # domain_pred_t = discriminator(target_intermediate)
            # adversarial_loss_t = adversarial_criterion(domain_pred_t, torch.zeros(domain_pred_t.size(0), dtype=torch.long, device=device))

            target_intermediate = target_extractor.forward(target_x)
            domain_pred_t = discriminator(target_intermediate)
            adversarial_loss_t = adversarial_criterion(domain_pred_t,
                                                       torch.zeros(domain_pred_t.size(0), dtype=torch.long,
                                                                   device=device))

            target_optimizer.zero_grad()
            adversarial_loss_t.backward()
            target_optimizer.step()

            # Metrics and loss accumulation
            total_loss = source_class_loss + adversarial_loss_s + adversarial_loss_t

            train_loss += total_loss.item()
            train_metrics += np.concatenate((
                np.array([source_class_loss.item(), 0, (adversarial_loss_s + adversarial_loss_t).item()]),
                source_metrics.compute(y_pred=source_pred, y_true=source_y),
                discriminator_metrics.compute(y_pred=torch.cat((domain_pred_s, domain_pred_t)), y_true=torch.cat((
                    torch.ones(domain_pred_s.size(0), dtype=torch.long, device=device),
                    torch.zeros(domain_pred_t.size(0), dtype=torch.long, device=device))))
            ))

            current_loss = train_loss / (i + 1)
            train_range.set_description(
                f"TRAIN -> epoch: {epoch} || loss: {current_loss:e} source: {source_class_loss.item():.2f} adv_s: {adversarial_loss_s.item():.2f} adv_t: {adversarial_loss_t.item():.2f}")
            train_range.refresh()


        ###############################################################
        # Validation                                                  #
        ###############################################################

        val_loss = 0
        val_metrics = np.zeros(len(metrics_name))
        val_range = tqdm(zip(source_val_loader, target_val_loader))

        source_model.eval()
        target_model.eval()
        discriminator.eval()
        with torch.no_grad():
            for i, (source_item, target_item) in enumerate(val_range):
                source_x = source_item['image'].to(device)
                source_y = source_item['label'].to(device)

                target_x = target_item['image'].to(device)

                # # Source CNN forward pass and classification loss
                # source_intermediate, source_pred = source_model.forward_and_get_intermediare(source_x)
                # source_class_loss = classification_criterion(source_pred, source_y)
                #
                # # Discriminator update (with source representation)
                # domain_pred_s = discriminator(source_intermediate)
                # adversarial_loss_s = adversarial_criterion(domain_pred_s, torch.ones(domain_pred_s.size(0), dtype=torch.long, device=device))
                #
                # # Target CNN update
                # target_intermediate, _ = target_model.forward_and_get_intermediare(target_x)
                # domain_pred_t = discriminator(target_intermediate)
                # adversarial_loss_t = adversarial_criterion(domain_pred_t, torch.zeros(domain_pred_t.size(0), dtype=torch.long, device=device))

                source_intermediate = source_extractor(source_x)
                source_pred = source_classifier(source_intermediate)
                source_class_loss = classification_criterion(source_pred, source_y)




                # Discriminator forward pass and adversarial loss for source data
                domain_pred_s = discriminator(source_intermediate)
                adversarial_loss_s = adversarial_criterion(domain_pred_s,
                                                           torch.ones(domain_pred_s.size(0), dtype=torch.long,
                                                                      device=device))

                # Target CNN forward pass and adversarial loss for target data
                # target_intermediate, _ = target_model.forward_and_get_intermediare(target_x)
                target_intermediate = target_extractor(target_x)
                domain_pred_t = discriminator(target_intermediate)
                adversarial_loss_t = adversarial_criterion(domain_pred_t, torch.zeros(domain_pred_t.size(0), dtype=torch.long, device=device))

                # Metrics and loss accumulation for validation
                total_loss = source_class_loss + adversarial_loss_s + adversarial_loss_t

                val_loss += total_loss.item()
                val_metrics += np.concatenate((
                    np.array([source_class_loss.item(), 0, (adversarial_loss_s + adversarial_loss_t).item()]),
                    source_metrics.compute(y_pred=source_pred, y_true=source_y),
                    discriminator_metrics.compute(y_pred=torch.cat((domain_pred_s, domain_pred_t)), y_true=torch.cat((
                        torch.ones(domain_pred_s.size(0), dtype=torch.long, device=device), torch.zeros(domain_pred_t.size(0), dtype=torch.long, device=device))))
                ))

                current_loss = val_loss / (i + 1)
                val_range.set_description(
                    f"VAL   -> epoch: {epoch} || loss: {current_loss:.2f} source: {source_class_loss.item():.2f} adv_s: {adversarial_loss_s.item():.2f} adv_t: {adversarial_loss_t.item():.2f}")
                val_range.refresh()

        ###################################################################
        # Save Scores in logs                                             #
        ###################################################################
        train_loss /= n_train
        val_loss /= n_val
        train_metrics /= n_train
        val_metrics /= n_val

        if save_experiment:
            train_step_logger(path=logging_path,
                              epoch=epoch,
                              train_loss=train_loss,
                              val_loss=val_loss,
                              train_metrics=train_metrics,
                              val_metrics=val_metrics)

            if val_loss < best_val_loss:
                print('Saving model weights')
                torch.save(source_model.get_classifier().state_dict(), os.path.join(logging_path, 'source_checkpoint.pt'))
                torch.save(target_model.get_feature_extractor().state_dict(), os.path.join(logging_path, 'target_checkpoint.pt'))
                # torch.save(discriminator.state_dict(), os.path.join(logging_path, 'discriminator_checkpoint.pt'))
                best_val_loss = val_loss

            print(f'{best_val_loss = }')

    stop_time = time.time()
    print(f"Training time: {stop_time - start_time} seconds for {config.learning.epochs} epochs")

    if save_experiment:
        plot_learning_curves.save_learning_curves(path=logging_path)


if __name__ == '__main__':
    import yaml

    config = EasyDict(yaml.safe_load(open('config/config.yaml')))
    config.model.name = 'adda'
    train(config=config)
