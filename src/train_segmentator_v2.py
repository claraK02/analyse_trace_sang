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
import matplotlib.pyplot as plt
import numpy as np

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
from torch import nn

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, output, target):
        # Apply sigmoid activation
        output = torch.sigmoid(output)
        
        intersection = (output * target).sum()
        union = output.sum() + target.sum()
        dice = 2.0 * intersection / (union + self.eps)
        return 1.0 - dice


def train(config: EasyDict, unet: UNet, classifier: Classifier, alpha: float = 0.5) -> None:
    # Get data
    train_generator = create_dataloader(config=config, mode='train')
    test_generator = create_dataloader(config=config, mode='test')
    n_train = len(train_generator)
    print(f"Found {n_train} training batches")

    # Loss
    #criterion_unet = torch.nn.BCEWithLogitsLoss()
    criterion_unet= DiceLoss()
    criterion_classif = torch.nn.CrossEntropyLoss()
    # Replace CrossEntropyLoss with KLDivLoss
    criterion_KL = torch.nn.KLDivLoss(reduction='batchmean')

    # Optimizer
    unet_optimizer = Adam(unet.parameters(), lr=0.001)
    classifier_optimizer = Adam(classifier.parameters(), lr=0.001)

    # Get and put on device
    device='cuda'
    unet.to(device) 
    classifier.to(device)

    #number of epochs
    epochs=5

    # Lists to store losses
    train_losses = []

    # Train unet and classifier on the data
    for epoch in tqdm(range(1, epochs+1)):
        train_loss = 0
        train_range = tqdm(train_generator)

        unet.train()
        classifier.train()
        for i, (x, y,back_true) in enumerate(train_range): #x est l'image, y est la classe de l'image, back_true est le background de l'image
            #print("shape de x",x.shape)
            #print("shape de y",y.shape)
            #print("shape de back_true",back_true.shape)
            segm_mask=batched_segmentation(x).long()
            x,back_true, y = x.to(device),back_true.to(device), y.to(device)

            # 1/ Predict the mask with the Unet
            pred = unet(x)

            segm_mask = segm_mask.to(device) # Move to device the image and the segmentation mask

            
            # 2/bis Use the predicted mask to extract the region of interest from the original image
            # The mask is binary (0 and 1), so we can use it as a multiplier to keep the region of interest and set the rest to 0
            roi = x * pred

            # 2/ Predict the background with the classifier exemple: [0.0,0.3,0.7,0.0]
            class_pred = classifier(roi)

            # 3/ Backpropagate the loss of the classifier in the classifier
            loss_classif = criterion_classif(class_pred, back_true)
            loss_classif.backward(retain_graph=True) 
            classifier_optimizer.step()
            classifier_optimizer.zero_grad()

            # 4/ Backpropagate the loss of the Unet - alpha * the loss of the classifier in the classifier then in the Unet without changing the weights of the classifier but updating those of the UNet
            loss_unet = criterion_unet(pred, segm_mask.float())

            #POSSIBILITE 1: On entraine le Unet a tromper le classifier sur le background qu'il prédit
            loss = loss_unet - alpha * loss_classif.detach()

            #POSSIBILITE 2: On entraine le Unet a tromper le classifier en le forçant à prédire la distribution uniforme des backgrounds (équivalent à ne pas savoir quel est le background)
            #loss =loss_unet +alpha * criterion_KL(class_pred,back_true)
            loss.backward()
            unet_optimizer.step()
            unet_optimizer.zero_grad()

            train_loss += loss.item()

            current_loss = train_loss / (i + 1)
            train_range.set_description(f"TRAIN -> epoch: {epoch} || loss: {current_loss:.4f}")
            train_range.refresh()

        train_losses.append(train_loss / n_train)

    def test_model(unet, test_dataloader, device='cuda'):
        unet.eval()  # Set the model to evaluation mode

        with torch.no_grad():  # No need to track gradients
            x,y,z = next(iter(test_dataloader))  # We take only the first batch
            x = x.to(device)  # Move the images to the device
            pred = unet(x)  # Apply the model
            pred = torch.sigmoid(pred)  # Apply sigmoid to get pixel probabilities
            pred = (pred > 0.5).float()  # Binarize predictions to 0 and 1

            # Move the images and predictions to cpu
            x = x.cpu().numpy()
            pred = pred.cpu().numpy()

            # Plot the original image and the segmentation
            for i in range(min(10, x.shape[0])):  # We take only the first 10 images
                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(np.transpose(x[i], (1, 2, 0)))  # Original image
                ax[1].imshow(np.transpose(pred[i], (1, 2, 0)))  # Segmentation
                plt.show()

    test_model(unet,test_generator)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__=="__main__":
    unet=UNet()
    classifier=Classifier(num_classes=4)
    config = EasyDict(yaml.safe_load(open('config/config.yaml')))  # Load config file
    train(config=config, unet=unet, classifier=classifier,alpha=0.3)  # Train the model