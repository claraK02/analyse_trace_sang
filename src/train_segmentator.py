import os
import sys
import time
import numpy as np
from tqdm import tqdm
from icecream import ic
from itertools import chain
from easydict import EasyDict
from os.path import dirname as up
import yaml
from explainable.create_mask import segment_image_file, plot_img_and_mask,batched_segmentation

import torch
from torch import Tensor
from torch.optim import Adam

#add one level above directory to the path

sys.path.append(up(up(__file__)))



from config.config import train_step_logger, train_logger
from src.dataloader.dataloader import create_dataloader
from metrics import Metrics
from src.model import resnet, adversarial,segmentation_model
from src.model.segmentation_model import UNet, Classifier
from utils import utils, plot_learning_curves


import matplotlib.pyplot as plt
from tqdm import tqdm

def train(config: EasyDict, unet: UNet, classifier: Classifier, k: float, pretrain_epochs: int) -> None:
    # Get data
    train_generator = create_dataloader(config=config, mode='train')
    val_generator = create_dataloader(config=config, mode='val')
    n_train, n_val = len(train_generator), len(val_generator) 
    print(f"Found {n_train} training batches and {n_val} validation batches")

    # Loss
    criterion = torch.nn.BCEWithLogitsLoss()

    # Optimizer and Scheduler
    unet_optimizer = Adam(unet.parameters(), lr=0.001)
    classifier_optimizer = Adam(classifier.parameters(), lr=0.001)

    # Get and put on device
    device='cuda'
    unet.to(device) 
    classifier.to(device)

    # Pretrain unet on k% of the data
    pretrain_batches = int(k * n_train)
    unet.train()

    # Lists to store losses
    pretrain_losses = []
    train_losses = []
    val_losses = []

    for epoch in range(pretrain_epochs):
        epoch_loss = 0
        train_range = tqdm(train_generator, total=len(train_generator))

        for i, (x, y,back_true) in enumerate(train_generator):
            if i >= pretrain_batches:
                break

            segm_mask=batched_segmentation(x).long()
            x,segm_mask = x.to(device), segm_mask.to(device) # Move to device the image and the segmentation mask
            pred = unet(x) # Forward pass with Unet
            loss = criterion(pred, segm_mask.float()) # Compute loss
            loss.backward()
            unet_optimizer.step()
            unet_optimizer.zero_grad()

            epoch_loss += loss.item()
            current_loss=epoch_loss/(i+1)
            train_range.set_postfix({'loss': current_loss})  # Update the progress bar with the current loss
            #train_range.set_description(f"TRAIN -> epoch: {epoch} || loss: {current_loss:.4f}")
            #train_range.refresh()



        pretrain_losses.append(epoch_loss / pretrain_batches)

    # Train unet and classifier on the remaining data
    for epoch in tqdm(range(1, config.learning.epochs + 1)):
        train_loss = 0
        train_range = tqdm(train_generator)

        unet.train()
        classifier.train()
        for i, (x, y,back_true) in enumerate(train_range):
            if i < pretrain_batches:
                continue
            
            x,back_true, y = x.to(device),back_true.to(device), y.to(device)
            pred = unet(x)
            class_pred = classifier(pred)
            loss = criterion(class_pred, y)
            loss.backward()
            classifier_optimizer.step()
            unet_optimizer.step()
            classifier_optimizer.zero_grad()
            unet_optimizer.zero_grad()
            train_loss += loss.item()

            current_loss = train_loss / (i + 1 - pretrain_batches)
            train_range.set_description(f"TRAIN -> epoch: {epoch} || loss: {current_loss:.4f}")
            train_range.refresh()

        train_losses.append(train_loss / (n_train - pretrain_batches))

        # Validation
        val_loss = 0
        val_range = tqdm(val_generator)

        unet.eval()
        classifier.eval()
        with torch.no_grad():
            for i, (x, y,back_true) in enumerate(val_range):
                x, y = x.to(device), y.to(device)
                pred = unet(x)
                class_pred = classifier(pred)
                loss = criterion(class_pred, y)
                val_loss += loss.item()

                current_loss = val_loss / (i + 1)
                val_range.set_description(f"VAL   -> epoch: {epoch} || loss: {current_loss:.4f}")
                val_range.refresh()

        val_losses.append(val_loss / n_val)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, pretrain_epochs + 1), pretrain_losses, label='Pretraining Loss')
    plt.plot(range(1, config.learning.epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, config.learning.epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__=="__main__":
    unet=UNet()
    classifier=Classifier(num_classes=10)
    config = EasyDict(yaml.safe_load(open('config/config.yaml')))  # Load config file
    train(config=config, unet=unet, classifier=classifier, k=0.8,pretrain_epochs=5)

