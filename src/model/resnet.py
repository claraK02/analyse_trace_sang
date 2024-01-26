import os
import sys
from os.path import dirname as up

import torch
from torch import nn, Tensor
from torchvision import models
from torchvision.models.resnet import ResNet18_Weights

sys.path.append(up(up(up(os.path.abspath(__file__)))))

from src.model.basemodel import Model


class Resnet(Model):
    def __init__(self,
                 num_classes: int,
                 ) -> None:   
        super(Resnet, self).__init__()

        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet_begin = nn.Sequential(*(list(resnet.children())[:-1]))

        for param in self.resnet_begin.parameters():
            param.requires_grad = False
        
        self.resnet_end = torch.nn.Linear(512, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        input: x is a tensor of shape (batch_size, 3, 128, 128)
        output: y is a tensor of shape (batch_size, num_classes)
        """
        x = self.resnet_begin(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.resnet_end(x)
        return x


if __name__ == '__main__':
    from torch import nn
    model = Resnet(num_classes=2)
    print("Total parameters:", model.get_number_parameters())
    print("Trainable parameters:", model.get_number_learnable_parameters())
    learnable_param = model.get_dict_learned_parameters()
    print('learnable parameters', learnable_param)
    model.load_dict_learnable_parameters(state_dict=learnable_param, strict=True)

    x = torch.randn((32, 3, 128, 128))
    y = model.forward(x)

    print("y shape:",y.shape)

