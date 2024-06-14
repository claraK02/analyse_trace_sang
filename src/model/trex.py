import os
import sys
from typing import Iterator
from easydict import EasyDict
from os.path import dirname as up
import torch
import torch.nn as nn
from torch import Tensor
from torchvision import models

sys.path.append(up(up(up(os.path.abspath(__file__)))))

from src.model.basemodel import Model


class Trex(Model):
    def __init__(self,
                 num_classes: int,
                 hidden_size: int,
                 p_dropout: float,
                 freeze_resnet: bool =True,
                 checkpoint_path: None =None, ) -> None:
        super(Trex, self).__init__()

        """
        Fine-tuned t-ReX model for classification.
        Args:
            num_classes (int): The number of output classes.
            hidden_size (int): The size of the hidden layer.
            p_dropout (float): The dropout probability.
            freeze_resnet (bool, optional): Whether to freeze the ResNet layers. Defaults to True.
            checkpoint_path (None | str, optional): Path to checkpoint. Defaults to None.
        """
        # Load the ResNet-50 model
        resnet = models.resnet50(weights = None)


        # Load checkpoint if provided
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            msg = resnet.load_state_dict(state_dict, strict=False)
            assert msg.missing_keys == ["fc.weight", "fc.bias"] and msg.unexpected_keys == []


        self.resnet_begin = nn.Sequential(*(list(resnet.children())[:-1]))

        if freeze_resnet:
            for param in self.resnet_begin.parameters():
                param.requires_grad = False
        self.resnet_begin.eval()

        self.fc1 = nn.Linear(2048, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=p_dropout)
        self.fc2 = nn.Linear(hidden_size, num_classes)


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

    def load_checkpoint(self, checkpoint_path):
         checkpoint = torch.load(checkpoint_path)
         state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
         self.load_state_dict(state_dict, strict=False)


    def forward_and_get_intermediare(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward pass of the model and returns intermediate and final outputs.

        Args:
            x (Tensor): Input tensor of shape (batch_size, 3, 128, 128).

        Returns:
            tuple[Tensor, Tensor]: A tuple containing:
                - intermediare (Tensor): Intermediate tensor of shape (batch_size, hidden_size).
                - reel_output (Tensor): Final output tensor of shape (batch_size, num_classes).
        """
        x = self.resnet_begin(x)
        x = x.squeeze(-1).squeeze(-1)
        intermediare = self.relu(self.fc1(x))
        x = self.dropout(intermediare)
        reel_output = self.fc2(x)
        return intermediare, reel_output

    def get_intermediare_parameters(self) -> Iterator[nn.Parameter]:
        """
        Get the intermediate parameters of the model, witch are the last two fully connected layers.

        Returns:
            An iterator over the intermediate parameters of the model.
        """
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



def get_trex(config: EasyDict) -> Trex:
    """Return a t-ReX model based on the given configuration.

        Args:
            config (EasyDict): The configuration object containing the model parameters.

        Returns:
            Trex: The instantiated t-ReX model.
        """
    trex = Trex(num_classes=config.data.num_classes,
                **config.model.trex)
    return trex


if __name__ == '__main__':
    import yaml
    import torch

    config_path = 'config/config.yaml'
    config = EasyDict(yaml.safe_load(open(config_path)))
    checkpoint_path = 'trex.pth'
    model = get_trex(config)

    # Print parameter counts
    print("Total parameters:", model.count_total_parameters())
    print("Trainable parameters:", model.count_trainable_parameters())
    learnable_param = model.get_dict_learned_parameters()
    #model.load_dict_learnable_parameters(state_dict=learnable_param, strict=True)

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

