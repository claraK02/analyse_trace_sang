import os
from PIL import Image
from easydict import EasyDict

from torch import Tensor
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class InferDataGenerator(Dataset):
    def __init__(self, datapath: str, image_size: int) -> None:
        """
        Initializes the InferDataGenerator class.

        Args:
        datapath: str, the path to the directory containing the images or the path to a single image file.
        image_size: int, the desired size of the images (both width and height).
        """
        self.data: list[str] = []

        if not is_image(datapath):
            for dirpath, _, filenames in os.walk(datapath):
                if filenames != []:
                    good_files = filter(is_image, filenames)
                    self.data += list(
                        map(lambda file: os.path.join(dirpath, file), good_files)
                    )
        else:
            self.data = [datapath]

        self.image_size = (image_size, image_size)
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        """
        Returns the total number of images in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[Tensor, str]:
        """
        Returns the image and its corresponding path at the given index.

        Args:
            index: int, the index of the image to load.

        Returns:
            (x, image_path): tuple[Tensor, str], a tuple containing the image tensor 
                and its corresponding path.
        """
        image_path = self.data[index]
        image = Image.open(image_path)
        x = self.transform(image)
        return x, image_path


def is_image(file: str) -> bool:
    """
    Check if a file is an image.

    Args:
        file (str): The file to check.

    Returns:
        bool: True if the file is an image, False otherwise.
    """
    return file.endswith(("jpeg", "png", "jpg", "JPEG", "PNG", "JPG"))


def create_infer_dataloader(config: EasyDict, datapath: str) -> DataLoader:
    """
    Create an inference dataloader.

    Args:
        config (EasyDict): Configuration object.
        datapath (str): Path to the data.

    Returns:
        DataLoader: Inference dataloader.
    """
    generator = InferDataGenerator(datapath, image_size=config.data.image_size)
    dataloader = DataLoader(
        dataset=generator,
        batch_size=min(config.test.batch_size, len(generator)),
        shuffle=False,
        drop_last=False,
        num_workers=config.test.num_workers,
    )

    return dataloader


if __name__ == "__main__":
    datapath = r"data\data_labo\test_512"
    infer_generator = InferDataGenerator(datapath=datapath, image_size=512)
    print(f"{len(infer_generator)=}")
    x, y = infer_generator.__getitem__(index=0)
    print(f"{x.shape=}")
