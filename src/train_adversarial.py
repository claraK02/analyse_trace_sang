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
from model.inceptionresnet import InceptionResNet,AdversarialInceptionResNet


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
    model = AdversarialInceptionResNet(num_classes=3)
    model = model.to(device)
    

    # Loss
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    # Optimizer and Scheduler

    # We define two optimizers: one for the main model and one for the adversary
    optimizer_main = torch.optim.Adam(model.main_parameters(), lr=config.learning.learning_rate_resnet)
    optimizer_adv = torch.optim.Adam(model.adversary_parameters(), lr=config.learning.learning_rate_adversary)

    
    #optimizer = torch.optim.Adam(model.parameters(), lr=config.learning.learning_rate)
    #scheduler = MultiStepLR(optimizer, milestones=config.learning.milestones, gamma=config.learning.gamma)

    # Save experiment
    save_experiment = config.learning.save_experiment #str true or false


    ###############################################################
    # Start Training                                              #
    ###############################################################
    start_time = time.time()
    best_val_loss = 0

    for epoch in range(1, config.learning.epochs + 1):
        print("epoch: ", epoch)
        train_loss = 0
        train_composed_loss = 0
        train_back_loss = 0
        train_metrics = 0
        background_metrics = 0
        train_range = tqdm(train_generator)

        # Training
        model.train()
        for i, (x, y_true,z) in enumerate(train_range): #ATTENTION: z is the background !

            x = x.to(device) #x shape: torch.Size([32, 3, 316, 316])
            y_true = y_true.to(device) #y_true shape: torch.Size([32])
            output = model.forward(x) #output shape: [torch.Size([32, 3]),torch.Size([32, 4])] ATTENTION: y_pred[0] is the prediction for the class, y_pred[1] is the prediction for the background
            y_pred = output[0] #y_pred shape: torch.Size([32, 3])
            z_pred = output[1] #back_pred shape: torch.Size([32, 4])
            z_true = z.to(device) #z_true shape: torch.Size([32])

            #print('y_pred_0',y_pred.shape)

            #Weight of the background loss
            alpha = 1

            
            #We first calculate the loss for the classes prediction
            main_loss = criterion(y_pred, y_true) #compares one hot vector and the indices vector

            #We then calculate the loss for the background prediction
            background_loss = criterion(z_pred, z_true) #compares one hot vector and the indices vector

            #We keep the values of the losses for the main model
            train_loss += main_loss.item()
            train_metrics += accuracy_pytorch(y_true=y_true, y_pred=y_pred).item() #we want a float
            background_metrics += accuracy_pytorch(y_true=z_true, y_pred=z_pred).item() #we want a float
            train_composed_loss +=(main_loss/background_loss).item() #(main_loss - alpha*background_loss).item()  #ATTENTION: we want to minimize this loss so to maximize the background loss so it is negative !!
            train_back_loss += background_loss.item()

            # Zero the gradients
            optimizer_main.zero_grad()
            optimizer_adv.zero_grad()

            # Backward pass and optimization of the main based on the main loss
            composed_loss = main_loss/background_loss                        #main_loss + alpha*background_loss
            composed_loss.backward(retain_graph=True)  # ATTENTION: we need to keep the graph for the adversary
            optimizer_main.step()

            background_loss.backward()  # We want to maximize this loss
            optimizer_adv.step()

            #get the current loss 
            current_loss = train_loss / (i + 1)
            current_composed_loss = train_composed_loss / (i + 1)
            current_back_loss = train_back_loss / (i + 1)


            #get the current metrics
            current_metrics = train_metrics / (i + 1)   
            current_background_metrics = background_metrics / (i + 1)

            #set the description of the progress bar
            train_range.set_description(f"TRAIN -> epoch: {epoch} || loss: {current_loss:.4f} || metrics: {current_metrics:.4f} || background metrics: {current_background_metrics:.4f} || background loss: {current_back_loss:.4f} || composed loss: {current_composed_loss:.4f}")
            train_range.refresh()

        ###############################################################
        # Start Validation                                            #
        ###############################################################


        val_loss = 0
        val_back_loss = 0
        val_composed_loss = 0
        val_metrics = 0
        val_background_metrics = 0
        val_range = tqdm(val_generator)

        model.eval()

        with torch.no_grad():

            for i, (x, y_true, z_true) in enumerate(val_range):  #ATTENTION: z_true is the background !
                x = x.to(device)
                y_true = y_true.to(device)
                z_true = z_true.to(device)

                output = model.forward(x)
                y_pred = output[0]
                z_pred = output[1]

                main_loss = criterion(y_pred, y_true)
                background_loss = criterion(z_pred, z_true)

                val_loss += main_loss.item()
                val_back_loss += background_loss.item()
                val_composed_loss += (main_loss/background_loss).item()
                 #(main_loss + alpha*background_loss).item()

                val_metrics += accuracy_pytorch(y_true=y_true, y_pred=y_pred).item() #we want a float
                val_background_metrics += accuracy_pytorch(y_true=z_true, y_pred=z_pred).item() #we want a float

                current_loss = val_loss / (i + 1)
                current_back_loss = val_back_loss / (i + 1)
                current_composed_loss = val_composed_loss / (i + 1)

                current_metrics = val_metrics / (i + 1)
                current_val_background_metrics = val_background_metrics / (i + 1)

                val_range.set_description(f"VAL   -> epoch: {epoch} || loss: {current_loss:.4f} || back_loss: {current_back_loss:.4f} || composed_loss: {current_composed_loss:.4f} || metrics: {current_metrics:.4f} || background metrics: {current_val_background_metrics:.4f}")
                val_range.refresh()
        
        #scheduler.step()       

        ###################################################################
        # Save Scores in logs                                             #
        ###################################################################
        train_loss = train_loss / n_train
        val_loss = val_loss / n_val
        train_metrics = train_metrics / n_train
        val_metrics = val_metrics / n_val
        train_composed_loss = train_composed_loss / n_train
        train_back_loss = train_back_loss / n_train
        background_metrics = background_metrics / n_train


        if save_experiment:
            logging_path = train_logger(config, metrics_name=["accuracy"])

        # if save_experiment=='true':
        #     train_step_logger(path=logging_path, 
        #                       epoch=epoch, 
        #                       train_loss=train_loss, 
        #                       val_loss=val_loss,
        #                       train_metrics=train_metrics,
        #                       val_metrics=val_metrics)
            
        if config.learning.save_checkpoint=='true' and val_loss < best_val_loss:
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