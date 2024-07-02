import os
import sys
from easydict import EasyDict
from os.path import dirname as up

from torch import nn, Tensor

sys.path.append(up(up(up(os.path.abspath(__file__)))))

from src.model.basemodel import Model


class AdversarialDANN(Model):
    def __init__(self,
                 hidden_size: int,
                 resnet_hidden_size: int,
                 p_dropout: float,
                 domain_number: int
                 ) -> None:
        """
        Initialize the Adversarial DANN model.
        2 fully connected layers with dropout and ReLU activation.

        Args:
            hidden_size (int): The size of the hidden layer.
            resnet_hidden_size (int): The size of the input layer.
            p_dropout (float): The dropout probability.
            domain_number (int): The number of domains classes.
        """
        super().__init__()
        self.fc1 = nn.Linear(in_features=resnet_hidden_size, out_features=hidden_size)
        self.dropout = nn.Dropout(p_dropout)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=domain_number)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the AdversarialDANN model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, resnet_hidden_size).

        Returns:
            Tensor: Output tensor of shape (batch_size, domain_number).
        """
        x = self.fc1(x)
        x = self.dropout(self.relu(x))
        x = self.fc2(x)
        return x


def get_adv(config: EasyDict) -> AdversarialDANN:
    """ return resnet according the configuration """
    adv = AdversarialDANN(domain_number=config.data.domain_number,
                            resnet_hidden_size=config.model.resnet.hidden_size,
                            **config.model.dann)
    return adv


if __name__ == '__main__':
    import yaml
    config_path = 'config/config.yaml'   
    config = EasyDict(yaml.safe_load(open(config_path)))

    model = get_adv(config)
    print(model)