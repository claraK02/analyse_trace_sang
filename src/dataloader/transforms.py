from easydict import EasyDict

from torchvision.transforms import (
    Compose,
    RandomRotation,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    ColorJitter,
    ToTensor
)


def get_transforms(transforms_config: EasyDict,
                   mode: str
                   ) -> Compose:
    """ Compose transfrom if mode==train"""
    transform = []

    if mode == 'train':
        if transforms_config.run_rotation:
            transform.append(RandomRotation(degrees=(0, 180)))

        if transforms_config.run_hflip:
            transform.append(RandomHorizontalFlip(p=0.5))
        
        if transforms_config.run_vflip:
            transform.append(RandomVerticalFlip(p=0.5))
        
        if sum(transforms_config.color.values()) != 0:
            transform.append(ColorJitter(**transforms_config.color))
    
    transform.append(ToTensor())
    return Compose(transform)