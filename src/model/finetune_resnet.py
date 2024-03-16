import os
import sys
from itertools import chain
from typing import Iterator
from easydict import EasyDict
from os.path import dirname as up

import torch
from torch import nn, Tensor
from torchvision import models
from torchvision.models.resnet import ResNet18_Weights

sys.path.append(up(up(up(os.path.abspath(__file__)))))

from src.model.basemodel import Model


class FineTuneResNet(Model):
    def __init__(self,
                 num_classes: int,
                 hidden_size: int,
                 p_dropout: float,
                 freeze_resnet: bool = True,
                 **kwargs
                 ) -> None:
        """
        Fine-tuned ResNet model for classification.

        Args:
            num_classes (int): The number of output classes.
            hidden_size (int): The size of the hidden layer.
            p_dropout (float): The dropout probability.
            freeze_resnet (bool, optional): Whether to freeze the ResNet layers. Defaults to True.
            **kwargs: Additional keyword arguments.

        """
        super(FineTuneResNet, self).__init__()

        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet_begin = nn.Sequential(*(list(resnet.children())[:-1]))

        if freeze_resnet:
            for param in self.resnet_begin.parameters():
                param.requires_grad = False
        self.resnet_begin.eval()

        self.fc1 = nn.Linear(in_features=512, out_features=hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=p_dropout)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, 3, 128, 128).

        Returns:
            Tensor: Output tensor of shape (batch_size, num_classes).

        """
        x = self.resnet_begin(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc1(x)
        x = self.dropout(self.relu(x))
        x = self.fc2(x)
        return x

    def forward_and_get_intermediare(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        input:
            - x is a tensor of shape (batch_size, 3, 128, 128)
        
        output:
            - intermediare is a tensor of shape (batch_size, 1000)
            - reel_output  is a tensor of shape (batch_size, num_classes) 
        """
        x = self.resnet_begin(x)
        x = x.squeeze(-1).squeeze(-1)
        intermediare = self.relu(self.fc1(x))
        x = self.dropout(intermediare)
        reel_output = self.fc2(x)
        return intermediare, reel_output

    def get_intermediare_parameters(self) -> Iterator[nn.Parameter]:
        """
        Get the intermediate parameters of the model, wicht are the last two fully connected layers.

        Returns:
            An iterator over the intermediate parameters of the model.
        """
        # return chain(self.fc1.parameters(), self.fc2.parameters())
        return self.fc1.parameters()

    def train(self) -> None:
        """
        Sets the dropout layer to training mode.
        """
        self.dropout = self.dropout.train()

    def eval(self) -> None:
        """
        Sets the dropout layer to evaluation mode.
        """
        self.dropout = self.dropout.eval()


def get_finetuneresnet(config: EasyDict) -> FineTuneResNet:
    """Return a FineTuneResNet model based on the given configuration.

    Args:
        config (EasyDict): The configuration object containing the model parameters.

    Returns:
        FineTuneResNet: The instantiated FineTuneResNet model.
    """
    resnet = FineTuneResNet(num_classes=config.data.num_classes,
                            **config.model.resnet)
    return resnet



if __name__ == '__main__':
    import yaml
    import torch
    config_path = 'config/config.yaml'   
    config = EasyDict(yaml.safe_load(open(config_path)))

    model = get_finetuneresnet(config)
    print("Total parameters:", model.get_number_parameters())
    print("Trainable parameters:", model.get_number_learnable_parameters())
    # learnable_param = model.get_dict_learned_parameters()
    # print('learnable parameters', learnable_param)
    # model.load_dict_learnable_parameters(state_dict=learnable_param, strict=True)

    x = torch.randn((32, 3, 128, 128))
    y = model.forward(x)

    print("y shape:", y.shape)

    intermediare, reel_output = model.forward_and_get_intermediare(x)
    print("intermediare shape:", intermediare.shape, type(intermediare))
    print("reel_output shape:", reel_output.shape, type(reel_output))

    inter_param = model.get_intermediare_parameters()
    print(inter_param, type(inter_param))
    for param in inter_param:
        print(param)
