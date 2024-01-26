import os
import time
import pandas as pd
from PIL import Image
from typing import Tuple
from easydict import EasyDict

import torch
from torch import Tensor
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


LABEL = ['1- Modèle Traces passives', "15- Modèle Modèle d'impact", '4- Modèle Transfert glissé']
BACKGROUND = ['carrelage', 'papier', 'bois', 'lino']


class DataGenerator(Dataset):
    def __init__(self, config: EasyDict, mode: str) -> None:
        if mode not in ['train', 'val', 'test']:
            raise ValueError(f"Error, expected mode is train, val, or test but found: {mode}")
        self.mode = mode

        data_item_path = os.path.join(config.data.path, f'{self.mode}_item.csv')
        if not os.path.exists(data_item_path):
            raise FileNotFoundError(f"File {data_item_path} wasn't found")
        self.data = pd.read_csv(data_item_path)
        
        self.transform = transforms.Compose([
            transforms.Resize((config.data.image_size, config.data.image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        '''
        input: index of the image to load
        output: tuple (x, label, background)
        -----
        SHAPE & DTYPE
        x:      (3, image_size, image_size)     torch.float32
        label:  (1)                             torch.float32
        backg:  (1)                             torch.float32
        '''
        _, image_path, label, background = self.data.loc[index]

        # Get image
        img = Image.open(image_path)
        x = self.transform(img)

        # Get label
        if label not in LABEL:
            raise ValueError(f"Expected label in LABEL but found {label}")
        label = torch.tensor(LABEL.index(label), dtype=torch.float32)

        # Get background
        if background not in BACKGROUND:
            raise ValueError(f"Expected backdround in BACKGROUND but found {background}")
        background = torch.tensor(BACKGROUND.index(background), dtype=torch.float32)

        return x, label, background


def create_dataloader(config: EasyDict, mode: str) -> DataLoader:
    generator = DataGenerator(config=config, mode=mode)
    dataloader = DataLoader(dataset=generator,
                            batch_size=config.learning.batch_size,
                            shuffle=config.learning.shuffle,
                            drop_last=config.learning.drop_last,
                            num_workers=config.learning.num_workers)

    return dataloader 


if __name__ == '__main__':
    import yaml
    import time
    from icecream import ic 

    config = EasyDict(yaml.safe_load(open('config/config.yaml')))
    config.learning.num_workers = 1

    dataloader = create_dataloader(config=config, mode='train')
    print(dataloader.batch_size)

    start_time = time.time()
    x, label, background = next(iter(dataloader))
    stop_time = time.time()
    print(f'time to load a batch: {stop_time - start_time:2f}s for a batchsize={config.learning.batch_size}')
    ic(x.shape, x.dtype)
    ic(label, label.shape, label.dtype)
    ic(background, background.shape, background.dtype)