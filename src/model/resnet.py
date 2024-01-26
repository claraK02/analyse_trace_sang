from typing import Iterator

import torch
from torch import nn, Tensor
from torchvision import models
from torch.nn import Parameter
from torchvision.models.resnet import ResNet18_Weights


class Resnet(nn.Module):
    def __init__(self,
                 num_classes: int,
                 dropout_probability: float=0.5,
                 hidden_size: int=100
                 ) -> None:   
        super(Resnet, self).__init__()

        self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        for param in self.resnet.parameters():
            param.requires_grad = False
        
        self.resnet.fc = torch.nn.Linear(512, num_classes)


    def forward(self, x: Tensor) -> Tensor:
        """
        input: x is a tensor of shape (batch_size, 3, 512, 512)
        output: y is a tensor of shape (batch_size, num_classes)
        """
        return self.resnet(x)
    
    def get_parameters(self) -> Iterator[Parameter]:
        return self.parameters()
    
    def get_learned_parameters(self) -> Iterator[Parameter]:
        for param in self.resnet.parameters():
            if param.requires_grad:
                yield param

    def count_total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 
    
    def get_only_learned_parameters(self) -> dict[str, Tensor]:
        """ get only the learned parameters """
        state_dict: dict[str, Tensor] = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                state_dict[name] = param

        return state_dict



    

if __name__ == '__main__':
    model = Resnet(num_classes=2)
    print("Total parameters:",model.count_total_parameters())
    print("Trainable parameters:",model.count_trainable_parameters())
    print(model.resnet)

    for param in model.get_learned_parameters():
        print(param)

    x = torch.randn((32, 3, 128, 128))
    y = model.forward(x)

    print("y shape:",y.shape)
    print("y:",y)

