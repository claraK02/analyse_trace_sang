import os
import sys
from PIL import Image
from typing import Literal
from easydict import EasyDict
from os.path import dirname as up

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

sys.path.append(up(up(up(os.path.abspath(__file__)))))

from src.dataloader.transforms import get_transforms
from src.dataloader.labels import LABELS, BACKGROUND


class DataGenerator(Dataset):
    """
    A custom dataset class for generating data for training, validation, or testing.
    """
    def __init__(self,
                 data_path: str,
                 mode: Literal['train', 'val', 'test'],
                 image_size: int,
                 use_background: bool,
                 transforms: EasyDict
                 ) -> None:
        """
        Initialize the DataLoader object.

        Args:
            data_path (str): The path to the data directory.
            mode (Literal['train', 'val', 'test']): The mode of the DataLoader. Must be one of 'train', 'val', or 'test'.
            image_size (int): The size of the images.
            use_background (bool): Whether to use background images.
            transforms (EasyDict): The transforms configuration.

        Raises:
            ValueError: If the mode is not one of 'train', 'val', or 'test'.
            FileNotFoundError: If the data_path or background folders are not found.
        """

        if mode not in ["train", "val", "test"]:
            raise ValueError(f"Error, expected mode is train, val, or test",
                             f" but found: {mode}")
        self.mode = mode
        self.use_background = use_background

        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"{data_path} wans't found. ",
                f"Make sure that you have run get_data_transform correctly",
                f"with the image_size={image_size}",
            )

        self.data: list[tuple[str, str, str]] = []
        for label in LABELS:
            # find images in background folder
            for background in BACKGROUND:
                folder = os.path.join(data_path, label, background)
                if os.path.exists(folder):
                    for image_name in os.listdir(folder):
                        image_path = os.path.join(folder, image_name)
                        if image_path.endswith(('.png', '.jpg', '.jpeg', '.JPG')):
                            self.data.append((image_path, label, background))
                elif self.use_background:
                    raise FileNotFoundError(f"{folder} wasn't found")
                
            # find images directly in the path
            folder = os.path.join(data_path, label)
            for image_name in os.listdir(folder):
                image_path = os.path.join(folder, image_name)
                if not os.path.exists(image_path) and not self.use_background:
                    raise FileNotFoundError(f"{folder} wasn't found")
                if image_path.endswith(('.png', '.jpg', '.jpeg', '.JPG')):
                    self.data.append((image_path, label, None))

        print(f"dataloader for {mode}, datapath: {data_path}, with {len(self.data)} images")

        self.transform = get_transforms(transforms_config=transforms,
                                        mode=mode)

    def __len__(self) -> int:
        """
        Returns the length of the data.
        """
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """
        Retrieves the item at the specified index.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            dict[str, Tensor]: A dictionary containing the retrieved item 
            with the following keys: 'image', 'label', and 'background'.

        Raises:
            ValueError: If the label or background is not found in the predefined lists.

        -----
        KEYS        SHAPE                           DTYPE
        image:      (3, image_size, image_size)     torch.float32
        label:      (1)                             torch.int64
        background: (1)                             torch.int64
        """
        image_path, label, background = self.data[index]

        item: dict[str, Tensor] = {}

        # Get image
        img = Image.open(image_path)
        item['image'] = self.transform(img)

        # Get label
        if label not in LABELS:
            raise ValueError(f"Expected label in LABEL but found {label}")
        item['label'] = torch.tensor(LABELS.index(label), dtype=torch.int64)

        # Get background
        if self.use_background:
            if background not in BACKGROUND:
                raise ValueError(f"Expected background in {BACKGROUND}",
                                 f" but found {background}")
            item['background'] = torch.tensor(BACKGROUND.index(background),
                                              dtype=torch.int64)

        return item


def create_dataloader(config: EasyDict,
                      mode: Literal['train', 'val', 'test'],
                      run_real_data: bool = False
                      ) -> DataLoader:
    """
    Create a DataLoader object for loading data.

    Args:
        config (EasyDict): Configuration object containing data and learning settings.
        mode (Literal['train', 'val', 'test']): Mode of operation ('train', 'val', 'test').
        run_real_data (bool, optional): Flag indicating whether to run with real data. Defaults to False.

    Returns:
        DataLoader: DataLoader object for loading data.
    """

    if not run_real_data:
        data_path = config.data.path
    else:
        data_path = config.data.real_data_path

    data_path = os.path.join(data_path , f"{mode}_{config.data.image_size}")

    generator = DataGenerator(
        data_path=data_path,
        mode=mode,
        image_size=config.data.image_size,
        use_background=(not (run_real_data or 'real' in config.data.path)),
        transforms=config.data.transforms
    )

    config_info: EasyDict = config.learning if mode != 'test' else config.test
    if len(generator) < config_info.batch_size:
        print(f'UserWarning: batchsize > num data !', end=' ')
        print(f'Change batch size to {config_info.batch_size} from {len(generator)}')
        config_info.batch_size = len(generator)

    dataloader = DataLoader(
        dataset=generator,
        batch_size=config_info.batch_size,
        shuffle=config_info.shuffle,
        drop_last=config_info.drop_last,
        num_workers=config_info.num_workers,
    )

    return dataloader


if __name__ == "__main__":
    import sys
    import yaml
    import time
    from os.path import dirname as up

    sys.path.append(up(up(up(os.path.abspath(__file__)))))
    from src.dataloader.show_batch import plot_batch

    config_path = "config/config.yaml"
    config = EasyDict(yaml.safe_load(open(config_path)))
    config.learning.num_workers = 1

    dataloader = create_dataloader(config=config, mode="test", run_real_data=True)
    print(f'{dataloader.batch_size = }')

    start_time = time.time()
    item: dict[str, Tensor] = next(iter(dataloader))
    stop_time = time.time()

    print(f"time to load a batch: {stop_time - start_time:2f}s ", end='')
    print(f"for a batchsize={dataloader.batch_size}")

    x = item['image']
    print('image:', x.shape, x.dtype)
    label = item['label']
    print('label:', label.shape, label.dtype)
    if 'background' in item:
        background = item['background']
        print('background:', background.shape, background.dtype)

    # plot_batch(x=x)
