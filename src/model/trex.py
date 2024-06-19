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
                 freeze_backbone: bool = True,
                 checkpoint_path: str = None) -> None:
        super(Trex, self).__init__()

        """
        Fine-tuned model for classification using a checkpoint.

        Args:
            num_classes (int): The number of output classes.
            hidden_size (int): The size of the hidden layer.
            p_dropout (float): The dropout probability.
            freeze_backbone (bool, optional): Whether to freeze the backbone layers. Defaults to True.
            checkpoint_path (str, optional): Path to checkpoint. Defaults to None.
        """
        # Load the backbone model resnet
        backbone = models.resnet50(weights=None)


        # Load checkpoint if provided
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            msg = backbone.load_state_dict(state_dict, strict=False)
            assert msg.missing_keys == ["fc.weight", "fc.bias"] and msg.unexpected_keys == []

        self.backbone_begin = nn.Sequential(*(list(backbone.children())[:-1]))

        if freeze_backbone:
            for param in self.backbone_begin.parameters():
                param.requires_grad = False
        self.backbone_begin.eval()

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
        x = self.backbone_begin(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc1(x)
        x = self.dropout(self.relu(x))
        x = self.fc2(x)
        return x

    def load_checkpoint(self, checkpoint_path):
         checkpoint = torch.load(checkpoint_path)
         state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
         self.load_state_dict(state_dict, strict=False)
    def forward_and_get_intermediate(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward pass of the model and returns intermediate and final outputs.

        Args:
            x (Tensor): Input tensor of shape (batch_size, 3, 128, 128).

        Returns:
            tuple[Tensor, Tensor]: A tuple containing:
                - intermediate (Tensor): Intermediate tensor of shape (batch_size, hidden_size).
                - final_output (Tensor): Final output tensor of shape (batch_size, num_classes).
        """
        x = self.backbone_begin(x)
        x = x.squeeze(-1).squeeze(-1)
        intermediate = self.relu(self.fc1(x))
        x = self.dropout(intermediate)
        reel_output = self.fc2(x)
        return intermediate, reel_output

    def get_intermediate_parameters(self) -> Iterator[nn.Parameter]:
        """
        Get the intermediate parameters of the model, which are the last two fully connected layers.

        Returns:
            An iterator over the intermediate parameters of the model.
        """
        return self.fc1.parameters()

    def train(self, mode=True) -> None:
        """
        Sets the model to training mode.
        """
        # super().train(mode)
        # self.dropout.train(mode)
        self.dropout = self.dropout.train()

    def eval(self) -> None:
        """
        Sets the model to evaluation mode.
        """
        # super().eval()
        # self.dropout.eval()
        self.dropout = self.dropout.eval()


def get_trex(config: EasyDict) -> Trex:
    """Return a Trex model based on the given configuration.

    Args:
        config (EasyDict): The configuration object containing the model parameters.

    Returns:
        Trex: The instantiated Trex model.
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
    #model.load_checkpoint(checkpoint_path)

    # Print parameter counts
    print("Total parameters:", sum(p.numel() for p in model.parameters()))
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    learnable_param = model.get_dict_learned_parameters()
    model.load_dict_learnable_parameters(state_dict=learnable_param, strict=True)
    x = torch.randn((32, 3, 128, 128))
    y = model.forward(x)
    print("y shape:", y.shape)
    intermediate, final_output = model.forward_and_get_intermediate(x)
