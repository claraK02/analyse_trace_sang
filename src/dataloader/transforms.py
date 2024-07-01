from easydict import EasyDict

from torchvision.transforms import (
    Compose,
    RandomRotation,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    ColorJitter,
    ToTensor, Grayscale,
)


def get_transforms(transforms_config: EasyDict, mode: str) -> Compose:
    """
    Compose transforms if mode==train.

    Args:
        transforms_config (EasyDict): Configuration for the transforms.
        mode (str): The mode of operation. Should be "train" or "test".

    Returns:
        Compose: A composed transform object.
    """
    transform = []



    if mode == "train":
        if transforms_config.run_rotation:
            transform.append(RandomRotation(degrees=(0, 180)))

        if transforms_config.run_hflip:
            transform.append(RandomHorizontalFlip(p=0.5))

        if transforms_config.run_vflip:
            transform.append(RandomVerticalFlip(p=0.5))

        if sum(transforms_config.color.values()) != 0:
            color_config: dict[str, float] = transforms_config.color
            kwars = dict(
                map(
                    lambda key: (key, get_colorjitter_parameter(color_config[key])),
                    color_config,
                )
            )
            transform.append(ColorJitter(**kwars))

    transform.append(ToTensor())
    return Compose(transform)


def get_colorjitter_parameter(value: float) -> int | tuple[float, float]:
    """
    Get the color jitter parameter based on the input value.

    Args:
        value (float): The input value.

    Returns:
        int or tuple[float, float]: The color jitter parameter. If the input value is greater than 0.5,
        it returns a tuple with the first element as 0.5 and the second element as the input value. Otherwise,
        it returns the input value itself.
    """
    return (0.5, value) if value > 0.5 else value
