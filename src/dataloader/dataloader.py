import os
import sys
import time
from PIL import Image
from typing import Tuple
from easydict import EasyDict
from os.path import dirname as up

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

sys.path.append(up(up(up(os.path.abspath(__file__)))))

from src.dataloader.transforms import get_transforms


LABEL = ['traces_passive', 'transfert_glisse']
BACKGROUND = ['carrelage', 'papier', 'bois', 'lino']


class DataGenerator(Dataset):
    def __init__(self, config: EasyDict, mode: str) -> None:
        if mode not in ['train', 'val', 'test']:
            raise ValueError(f"Error, expected mode is train, val, or test but found: {mode}")
        self.mode = mode

        dst_path = os.path.join(config.data.path, f'{mode}_{config.data.image_size}')
        if not os.path.exists(dst_path):
            raise FileNotFoundError(f"{dst_path} wans't found. Make sure that you have run get_data_transform",
                                    f"with the image_size={config.data.image_size}")
        
        self.data: Tuple[str, str, str] = []
        for label in LABEL:
            for background in BACKGROUND:
                folder = os.path.join(dst_path, label, background)
                if not os.path.exists(folder):
                    raise FileNotFoundError(f"{folder} wasn't found")
                for image_name in os.listdir(folder):
                    self.data.append((os.path.join(folder, image_name), label, background))
        
        self.transform = get_transforms(transforms_config=config.data.transforms)
        print(self.transform)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        '''
        input: index of the image to load
        output: tuple (x, label, background)
        -----
        SHAPE & DTYPE
        x:      (3, image_size, image_size)     torch.float32
        label:  (1)                             torch.int64
        backg:  (1)                             torch.float32
        '''
        image_path, label, background = self.data[index]

        # Get image
        img = Image.open(image_path)
        x = self.transform(img)

        # Get label
        if label not in LABEL:
            raise ValueError(f"Expected label in LABEL but found {label}")
        label = torch.tensor(LABEL.index(label), dtype=torch.float32).long()

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
    import sys
    import yaml
    import time
    from icecream import ic 
    from os.path import dirname as up

    sys.path.append(up(up(up(os.path.abspath(__file__)))))
    from src.dataloader.show_batch import plot_batch

    config_path = 'config/config.yaml'   
    config = EasyDict(yaml.safe_load(open(config_path)))
    config.learning.num_workers = 1

    # generator = DataGenerator(config=config, mode='train')
    # print(len(generator))
    # x, label, background = generator.__getitem__(index=3)
    # ic(x.shape, x.dtype)
    # ic(label, label.shape, label.dtype)
    # ic(background, background.shape, background.dtype)

    dataloader = create_dataloader(config=config, mode='train')
    print(dataloader.batch_size)

    start_time = time.time()
    x, label, background = next(iter(dataloader))
    stop_time = time.time()
    print(f'time to load a batch: {stop_time - start_time:2f}s for a batchsize={config.learning.batch_size}')
    ic(x.shape, x.dtype)
    ic(label, label.shape, label.dtype)
    ic(background, background.shape, background.dtype)

    plot_batch(x=x)