import os
import time
import numpy as np
from tqdm import tqdm
import yaml
from metrics import accuracy_one_hot, accuracy_pytorch

#add config to module path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.config import train_step_logger, train_logger

from easydict import EasyDict

import torch
from torch.optim.lr_scheduler import MultiStepLR
from dataloader.dataloader import create_image_classification_dataloader
from model.inceptionresnet import InceptionResNet


def train(config: EasyDict) -> None:

    # Use gpu or cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Get data
    train_generator = create_image_classification_dataloader(config=config, mode='train')
    val_generator = create_image_classification_dataloader(config=config, mode='val')
    n_train, n_val = len(train_generator), len(val_generator) 
    print(f"Found {n_train} training batches and {n_val} validation batches")

    # Get model
    model = InceptionResNet(num_classes=3)
    model = model.to(device)
    

    # Loss
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    # Optimizer and Scheduler
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning.learning_rate)
    scheduler = MultiStepLR(optimizer, milestones=config.learning.milestones, gamma=config.learning.gamma)

    # Save experiment
    save_experiment = config.learning.save_experiment #str true or false


    ###############################################################
    # Start Training                                              #
    ###############################################################
    start_time = time.time()

    for epoch in range(1, config.learning.epochs + 1):
        print("epoch: ", epoch)
        train_loss = 0
        train_metrics = 0
        train_range = tqdm(train_generator)

        # Training
        model.train()
        for i, (x, y_true) in enumerate(train_range):
            x = x.to(device) #x shape: torch.Size([32, 3, 316, 316])
            y_true = y_true.to(device) #y_true shape: torch.Size([32])
            y_pred = model.forward(x) #y_pred shape: torch.Size([32, 3])

            loss = criterion(y_pred, y_true) #compares one hot vector and the indices vector

            train_loss += loss.item()
            train_metrics += accuracy_pytorch(y_true=y_true, y_pred=y_pred).item() #we want a float

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            current_loss = train_loss / (i + 1)
            current_metrics = train_metrics / (i + 1)   
            train_range.set_description(f"TRAIN -> epoch: {epoch} || loss: {current_loss:.4f} || metrics: {current_metrics:.4f}")
            train_range.refresh()

        ###############################################################
        # Start Validation                                            #
        ###############################################################

        val_loss = 0
        #val_metrics = np.zeros((len(metrics_name)))
        val_range = tqdm(val_generator)

        model.eval()

        with torch.no_grad():

            val_loss = 0
            val_metrics = 0
            
            for i, (x, y_true) in enumerate(val_range):
                x = x.to(device)
                y_true = y_true.to(device)

                y_pred = model.forward(x)

                loss = criterion(y_pred, y_true)

                val_loss += loss.item()
                val_metrics += accuracy_pytorch(y_true=y_true, y_pred=y_pred).item() #we want a float

                current_loss = val_loss / (i + 1)
                current_metrics = val_metrics / (i + 1)
                val_range.set_description(f"VAL   -> epoch: {epoch} || loss: {current_loss:.4f} || metrics: {current_metrics:.4f}")
                val_range.refresh()
        
        scheduler.step()       

        ###################################################################
        # Save Scores in logs                                             #
        ###################################################################
        train_loss = train_loss / n_train
        val_loss = val_loss / n_val
        train_metrics = train_metrics / n_train
        val_metrics = val_metrics / n_val

        if save_experiment:
            logging_path = train_logger(config, metrics_name=["accuracy"])

        # if save_experiment=='true':
        #     train_step_logger(path=logging_path, 
        #                       epoch=epoch, 
        #                       train_loss=train_loss, 
        #                       val_loss=val_loss,
        #                       train_metrics=train_metrics,
        #                       val_metrics=val_metrics)
            
        if config.learning.save_checkpoint=='true':
            print('saving model weights...' )
            torch.save(model.state_dict(), os.path.join(logging_path, 'checkpoint.pt'))
            best_val_loss = val_loss
            print(f"best val loss: {best_val_loss:.4f}")

    stop_time = time.time()
    print(f"training time: {stop_time - start_time}secondes for {config.learning.epochs} epochs")
    print(f"Loss: {train_loss:.4f} (train) - {val_loss:.4f} (val)" - f"Accuracy: {train_metrics:.4f} (train) - {val_metrics:.4f} (val)")
    
    # if save_experiment:
    #     save_learning_curves(path=logging_path)


if __name__ == '__main__':
    config = EasyDict(yaml.safe_load(open('config/config.yaml')))  # Load config file
    train(config=config)