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
from src.dataloader.labels import LABELS, BACKGROUND, DOMAIN


class DataGenerator(Dataset):
    """
    A custom dataset class for generating data for training, validation, or testing.
    """
    def __init__(self,
                 data_path: str,
                 mode: Literal['train', 'val', 'test'],
                 use_background: bool,
                 transforms: EasyDict,
                 real_data_test: bool
                 ) -> None:
        """
        Initialize the DataLoader object.

        Args:
            data_path (str): The path to the data directory.
            mode (Literal['train', 'val', 'test']): The mode of the DataLoader. Must be one of 'train', 'val', or 'test'.
            use_background (bool): Whether to use background images.
            transforms (EasyDict): The transforms configuration.
            real_data_test (bool): Flag indicating whether to use real data for testing.

        Raises:
            ValueError: If the mode is not one of 'train', 'val', or 'test'.
            FileNotFoundError: If the data_path or background folders are not found.
        """
        print("data_path : ", data_path)
        if mode not in ["train", "val", "test"]:
            raise ValueError(f"Expected mode is 'train', 'val', or 'test', but found: {mode}")
        self.mode = mode
        self.use_background = use_background

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"{data_path} wasn't found.")

        self.data: list[tuple[str, str, str, str]] = []
        if self.mode in ["train", "val"]:
            for domain in ["data_labo", "new_real_data"]:
                print("data_path : ", data_path)
                domain_path = os.path.join(data_path, domain)
                if not os.path.exists(domain_path):
                    raise FileNotFoundError(f"{domain_path} wasn't found.")
                if domain == "new_real_data":
                    for label in LABELS:
                        folder = os.path.join(domain_path, mode + "_256", label)
                        if os.path.exists(folder):
                            for image_name in os.listdir(folder):
                                image_path = os.path.join(folder, image_name)
                                if image_path.endswith(('.png', '.jpg', '.jpeg', '.JPG')):
                                    self.data.append((image_path, label, None, domain))
                        elif self.use_background:
                            raise FileNotFoundError(f"{folder} wasn't found")
                else:
                    for label in LABELS:
                        for background in BACKGROUND:
                            folder = os.path.join(domain_path, mode + "_256", label, background)
                            if os.path.exists(folder):
                                for image_name in os.listdir(folder):
                                    image_path = os.path.join(folder, image_name)
                                    if image_path.endswith(('.png', '.jpg', '.jpeg', '.JPG')):
                                        self.data.append((image_path, label, None, domain))
                            elif self.use_background:
                                raise FileNotFoundError(f"{folder} wasn't found")

                        folder = os.path.join(domain_path, mode + "_256", label)
                        for image_name in os.listdir(folder):
                            image_path = os.path.join(folder, image_name)
                            if not os.path.exists(image_path) and not self.use_background:
                                raise FileNotFoundError(f"{folder} wasn't found")
        else:
            if real_data_test:
                domain = 'new_real_data'
                domain_path = os.path.join(data_path, domain)
                print("data_path : ", data_path)
                if not os.path.exists(domain_path):
                    raise FileNotFoundError(f"{domain_path} wasn't found.")
                for label in LABELS:
                    folder = os.path.join(domain_path, mode + "_256", label)
                    if os.path.exists(folder):
                        for image_name in os.listdir(folder):
                            image_path = os.path.join(folder, image_name)
                            if image_path.endswith(('.png', '.jpg', '.jpeg', '.JPG')):
                                self.data.append((image_path, label, None, domain))
                    elif self.use_background:
                        raise FileNotFoundError(f"{folder} wasn't found")
            else:
                domain = 'data_labo'
                domain_path = os.path.join(data_path, domain)
                print("data_path : ", data_path)
                if not os.path.exists(domain_path):
                    raise FileNotFoundError(f"{domain_path} wasn't found.")

                for label in LABELS:
                    for background in BACKGROUND:
                        folder = os.path.join(domain_path, mode + "_256", label, background)
                        if os.path.exists(folder):
                            for image_name in os.listdir(folder):
                                image_path = os.path.join(folder, image_name)
                                if image_path.endswith(('.png', '.jpg', '.jpeg', '.JPG')):
                                    self.data.append((image_path, label, None, domain))
                        elif self.use_background:
                            raise FileNotFoundError(f"{folder} wasn't found")

        print(f"Dataloader for {mode}, datapath: {data_path}, with {len(self.data)} images")
        self.transform = get_transforms(transforms_config=transforms, mode=mode)

    def __len__(self) -> int:
        """
        Returns the length of the data.

        Returns:
            int: The length of the data.
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
        """
        image_path, label, background, domain = self.data[index]

        item: dict[str, Tensor] = {}

        # Get image
        img = Image.open(image_path)
        item['image'] = self.transform(img)

        # Get label
        if label not in LABELS:
            raise ValueError(f"Expected label in LABELS but found {label}")
        item['label'] = torch.tensor(LABELS.index(label), dtype=torch.int64)

        # Get background
        if self.use_background:
            if background not in BACKGROUND:
                raise ValueError(f"Expected background in BACKGROUND but found {background}")
            item['background'] = torch.tensor(BACKGROUND.index(background), dtype=torch.int64)

        if domain not in DOMAIN:
            raise ValueError(f"Expected domain in DOMAIN but found {domain}")
        item['domain'] = torch.tensor(DOMAIN.index(domain), dtype=torch.int64)

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
        data_path=config.data.path,
        mode=mode,
        use_background=False,
        transforms=config.data.transforms,
        real_data_test=run_real_data
    )

    config_info: EasyDict = config.learning if mode != 'test' else config.test
    if len(generator) < config_info.batch_size:
        print(f'UserWarning: batchsize > num data !', end=' ')
        print(f'Change batch size to {len(generator)} from {config_info.batch_size}')
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
    import yaml
    import time
    from src.dataloader.show_batch import plot_batch

    config_path = "config/config.yaml"
    config = EasyDict(yaml.safe_load(open(config_path)))
    config.learning.num_workers = 1

    dataloader = create_dataloader(config=config, mode="test", run_real_data=True)

    print(f'{dataloader.batch_size = }')
    start_time = time.time()
    item: dict[str, Tensor] = next(iter(dataloader))
    stop_time = time.time()

    print(f"time to load a batch: {stop_time - start_time:.2f}s ", end='')
    print(f"for a batchsize={dataloader.batch_size}")

    x = item['image']
    print('image:', x.shape, x.dtype)
    label = item['label']
    print('label:', label.shape, label.dtype)
    if 'background' in item:
        background = item['background']
        print('background:', background.shape, background.dtype)
    print('domain:', item['domain'].shape, item['domain'].dtype)

    # plot_batch(x=x)
