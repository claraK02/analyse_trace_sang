import os
import sys
import random
import numpy as np
from PIL import Image
from easydict import EasyDict
from typing import Any, Literal
from os.path import dirname as up

import torch
from torch import Tensor
from torch.nn import Parameter
from torchvision import transforms

sys.path.append(up(os.path.abspath(__file__)))

from src.metrics.metrics import Metrics


def get_device(device_config: str) -> torch.device:
    """
    Get the device to be used for computation.

    Args:
        device_config (str): The desired device configuration. Valid values are 'cuda' or 'cpu'.

    Returns:
        torch.device: The device to be used for computation. It will be either 'cuda' if CUDA is available and 'cuda' is specified in device_config, or 'cpu' otherwise.
    """
    if torch.cuda.is_available() and device_config == 'cuda':
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def put_on_device(device: torch.device, *args: Any) -> None:
    """
    Put all arguments on the specified device.

    Args:
        device (torch.device): The device to move the arguments to.
        *args (Any): The arguments to be moved to the device.
    """
    for arg in args:
        arg = arg.to(device)


def get_metrics_name_for_adv(resnet_metrics: Metrics,
                             adv_metrics: Metrics
                             ) -> list[str]:
    """ get all metrics name

    Args:
        resnet_metrics (Metrics): The metrics for the resnet model.
        adv_metrics (Metrics): The metrics for the adv model.

    Returns:
        list[str]: A list of all metrics names.
    """
    add_name = lambda model_name, metric_name: f'{model_name}_{metric_name}'
    add_resnet = lambda metric_name: add_name(model_name='resnet', metric_name=metric_name)
    add_adv = lambda metric_name: add_name(model_name='adv', metric_name=metric_name)
    metrics_name = list(map(add_resnet, resnet_metrics.get_names())) + \
                   list(map(add_adv, adv_metrics.get_names()))
    return metrics_name


def load_weights(logging_path: str,
                 model_name: str = 'res',
                 device: torch.device = torch.device('cpu'),
                 endfile: str = '.pt'
                 ) -> dict[str, Parameter]:
    """
    Load weights from the specified logging path for a given model.

    Args:
        logging_path (str):
            The path where the weight files are stored.
        model_name (str, optional):
            The name of the model. Defaults to 'res'.
        device (torch.device, optional):
            The device to load the weights onto. Defaults to torch.device('cpu').
        endfile (str, optional):
            The file extension of the weight files. Defaults to '.pt'.

    Returns:
        dict[str, Parameter]:
            A dictionary containing the loaded weights.

    """
    weight_files = list(filter(lambda x: x.endswith(endfile),
                               os.listdir(logging_path)))

    if len(weight_files) == 1:
        weight = torch.load(os.path.join(logging_path, weight_files[0]),
                            map_location=device)
        return weight

    model_name_files = list(filter(lambda x: model_name in x, weight_files))

    if len(model_name_files) > 1:
        raise FileExistsError(f'Confused by multiple weights for the {model_name} model')
    if len(model_name_files) < 1:
        raise FileNotFoundError(f'No weights was found in {logging_path} with the name {model_name}')
    weight = torch.load(os.path.join(logging_path, model_name_files[0]),
                        map_location=device)
    return weight


def get_random_img(data_path: str = 'data/data_labo/test_256',
                   image_type: Literal['numpy', 'torch'] = 'torch'
                   ) -> tuple[np.ndarray | Tensor, str]:
    """Get a random image from the test dataset.

    Args:
        data_path (str): The path to the test dataset directory. Defaults to 'data/data_labo/test_256'.
        image_type (Literal['numpy', 'torch']): The type of image to return. Defaults to 'torch'.

    Returns:
        tuple[np.ndarray | Tensor, str]: A tuple containing the random image and its label.
    """
    label = random.choice(os.listdir(data_path))
    background = random.choice(os.listdir(os.path.join(data_path, label)))
    folder_path = os.path.join(data_path, label, background)
    images = os.listdir(folder_path)

    if len(images) == 0:
        # if there are not image, find an other one
        return get_random_img(data_path)

    image_path = os.path.join(folder_path, random.choice(images))
    print(f'{image_path=}')
    image = Image.open(image_path)

    if image_type == 'numpy':
        image = np.array(image)
    if image_type == 'torch':
        transform = transforms.Compose([transforms.ToTensor()])
        image = transform(image)

    return image, label


def convert_tensor_to_rgb(image: Tensor, normelize: bool = False) -> np.ndarray[float]:
    """Converts a torch tensor into a numpy array and rearranges the dimensions to (256, 256, 3).

    Args:
        image (Tensor): The input torch tensor.
        normelize (bool, optional): Whether to normalize the output array. Defaults to False.

    Returns:
        np.ndarray[float]: The converted numpy array.
    """
    rgb_img: np.ndarray = image.permute(1, 2, 0).numpy()
    rgb_img = rgb_img.astype(np.float32)
    if normelize:
        rgb_img = normalize_image(rgb_img)
    return rgb_img


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize the input image by scaling it to the range [0, 1].

    Args:
        image (np.ndarray): The input image to be normalized.

    Returns:
        np.ndarray: The normalized image.
    """
    return (image - image.min()) / (image.max() - image.min())


def resume_training(config: EasyDict, model: torch.nn.Module) -> None:
    """
    Resume training from a checkpoint.

    Args:
        config (EasyDict): The configuration object.
        model (torch.nn.Module): The model to resume training on.
    """
    if 'resume_training' not in config.model.resnet.keys():
        print("didn't find resume_training key")
        return None

    resume_training: EasyDict = config.model.resnet.resume_training
    if not resume_training.do_resume:
        print("resume training: None")
        return None

    print(f'resume training from {resume_training.path}')
    weight = load_weights(resume_training.path)
    model.load_dict_learnable_parameters(state_dict=weight, strict=True)
    del weight

    if not resume_training.freeze_param:
        print('unfreezing the parameters')
        for param in model.parameters():
            param.requires_grad = True

    return None


def get_relatif_image_path(image_path: str,
                           dst_path: str
                           ) -> str:
    """
    Get the relative image path by replacing the destination path with an empty string.

    Args:
        image_path (str): The absolute path of the image.
        dst_path (str): The destination path to be replaced.

    Returns:
        str: The relative image path.
    """
    return image_path.replace(dst_path, '')