import os
import sys
from os.path import dirname as up

from torch import nn, Tensor

sys.path.append(up(up(up(os.path.abspath(__file__)))))

from src.model.basemodel import Model


class AdversarialResNet(Model):
    def __init__(self,
                 hidden_size: int,
                 resnet_hidden_size: int,
                 p_dropout: float,
                 background_classes: int
                 ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features=resnet_hidden_size, out_features=hidden_size)
        self.dropout = nn.Dropout(p_dropout)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=background_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        input: x is a tensor of shape (batch_size, resnet_hidden_size)
        output: y is a tensor of shape (batch_size, background_classes)
        """
        x = self.fc1(x)
        x = self.dropout(self.relu(x))
        x = self.fc2(x)
        return x
