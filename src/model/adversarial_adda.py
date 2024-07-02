import os
import sys

import torch
from easydict import EasyDict
from os.path import dirname as up

from torch import nn, Tensor

sys.path.append(up(up(up(os.path.abspath(__file__)))))

from src.model.finetune_resnet import FineTuneResNet

class AdversarialDiscriminator(nn.Module):
    def __init__(self, hidden_size: int, resnet_hidden_size: int, p_dropout: float, domain_number: int) -> None:
        """
        Initialize the Adversarial Discriminator model.
        2 fully connected layers with dropout and ReLU activation.

        Args:
            hidden_size (int): The size of the hidden layer.
            resnet_hidden_size (int): The size of the input layer.
            p_dropout (float): The dropout probability.
            domain_number (int): The number of domain classes.
        """
        super().__init__()
        self.fc1 = nn.Linear(in_features=resnet_hidden_size, out_features=hidden_size)
        self.dropout = nn.Dropout(p_dropout)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=domain_number)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Adversarial Discriminator model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, resnet_hidden_size).

        Returns:
            Tensor: Output tensor of shape (batch_size, domain_number).
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def get_discriminator(config: EasyDict) -> AdversarialDiscriminator:
    """Return Adversarial Discriminator model based on the given configuration."""
    discriminator = AdversarialDiscriminator(
        domain_number=config.data.domain_number,
        resnet_hidden_size=config.model.resnet.hidden_size,
        **config.model.adda
    )
    return discriminator

def get_source_cnn(config: EasyDict) -> FineTuneResNet:
    """Return Source CNN model based on the given configuration."""
    source_cnn = FineTuneResNet(
        num_classes=config.data.num_classes,
        hidden_size=config.model.resnet.hidden_size,
        p_dropout=config.model.resnet.p_dropout,
        freeze_resnet=True
    )
    return source_cnn

def get_target_cnn(config: EasyDict) -> FineTuneResNet:
    """Return Target CNN model based on the given configuration."""
    target_cnn = FineTuneResNet(
        num_classes=config.data.num_classes,
        hidden_size=config.model.resnet.hidden_size,
        p_dropout=config.model.resnet.p_dropout,
        freeze_resnet=True
    )
    return target_cnn

# def get_source_cnn_feature_extractor(config: EasyDict) -> nn.Module:
#     """Return the feature extractor part of the Source CNN model based on the given configuration."""
#     source_cnn = get_source_cnn(config)
#     return source_cnn.get_feature_extractor()
#
# def get_source_cnn_classifier(config: EasyDict) -> nn.Module:
#     """Return the classifier part of the Source CNN model based on the given configuration."""
#     source_cnn = get_source_cnn(config)
#     return source_cnn.get_classifier()
#
# def get_target_cnn_feature_extractor(config: EasyDict) -> nn.Module:
#     """Return the feature extractor part of the Source CNN model based on the given configuration."""
#     target_cnn = get_target_cnn(config)
#     return target_cnn.get_feature_extractor()
#
# def get_target_cnn_classifier(config: EasyDict) -> nn.Module:
#     """Return the classifier part of the Source CNN model based on the given configuration."""
#     target_cnn = get_target_cnn(config)
#     return target_cnn.get_classifier()


if __name__ == '__main__':
    import yaml
    import torch
    from torchsummary import summary

    config_path = 'config/config.yaml'
    config = EasyDict(yaml.safe_load(open(config_path)))

    discriminator = get_discriminator(config)
    summary(discriminator, input_size=(1, 2048))

    source_cnn = get_source_cnn(config)
    summary(source_cnn, input_size=(3, 128, 128))

    target_cnn = get_target_cnn(config)
    summary(target_cnn, input_size=(3, 128, 128))

    source_cnn_feature_extractor = get_source_cnn_feature_extractor(config)
    print(source_cnn_feature_extractor)

    source_cnn_classifier = get_source_cnn_classifier(config)
    print(source_cnn_classifier)
