import os
import sys
from tqdm import tqdm
from typing import Callable
from easydict import EasyDict
from os.path import dirname as up

import torch
from torch import Tensor

sys.path.append(up(up(os.path.abspath(__file__))))

from src.dataloader.labels import get_topk_prediction
from src.dataloader.infer_dataloader import create_infer_dataloader
from src.model import finetune_resnet
from src.gradcam import GradCam
from utils import utils


def infer(infer_images_path: list[str],
          infer_datapath: str,
          logging_path: str,
          config: EasyDict,
          plot_saliency: bool,
          dstpath: str,
          filename: str,
          run_temperature_optimization: bool = True,
          sep: str = ','
          ) -> None:
    """
    Perform inference using the provided dataloader and model.

    Args:
        infer_dataloader (DataLoader): The dataloader for inference (can be None if infer_datapath is specify).
        infer_datapath (str): The path to the inference data (only in the case of infer_dataloader is None).
        logging_path (str): The path to the logging directory.
        config (EasyDict): The configuration object (most of times, in the logging_path).
        dstpath (str): The destination path for saving the inference results.
        filename (str): The filename for saving the inference results.
        run_temperature_optimization (bool, optional): Whether to run temperature optimization. Defaults to True.
        sep (str, optional): The separator for saving the inference results. Defaults to ','.

    Raises:
        ValueError: If both infer_dataloader and infer_datapath are None.
    """
    device = utils.get_device(device_config=config.learning.device)
    if device.type == 'cpu':
        config.test.batch_size = 32

    infer_dataloader = create_infer_dataloader(config=config,
                                               data=infer_images_path,
                                               datapath=infer_datapath)

    # Get model
    model = finetune_resnet.get_finetuneresnet(config)
    weight = utils.load_weights(logging_path, device=device, model_name='res')
    model.load_dict_learnable_parameters(state_dict=weight, strict=True)
    model = model.to(device)
    del weight

    get_image_name: Callable[[str], str] = lambda img_name: img_name.split(os.sep)[-1]
    temperature: float = 1.5
    output: list[list[tuple[int, str, float]]] = []
    image_names: list[str] = []

    # GradCAM
    if plot_saliency:
        gradcam = GradCam(model=model)
        saliency_path = os.path.join(dstpath, 'saliency_maps')
        saliency_fun_name = lambda img_name: get_image_name(img_name).split('.')[0] + '_saliency.png'

    model.eval()
    # with torch.no_grad():
    for x, image_path in tqdm(infer_dataloader, desc='Infering'):
        image_name = list(map(get_image_name, image_path))
        x: Tensor = x.to(device)
        y_pred = model.forward(x)

        if plot_saliency:
            visualizations = gradcam.forward(x)
            gradcam.save_saliency_maps(visualizations=visualizations,
                                       dstpath=saliency_path,
                                       filenames=list(map(saliency_fun_name, image_name)))

        if run_temperature_optimization:
            y_pred = torch.nn.functional.softmax(y_pred / temperature, dim=-1)
        else:
            y_pred = torch.nn.functional.softmax(y_pred, dim=-1)

        output += get_topk_prediction(y_pred, k=3)
        image_names += image_name
        
    save_infer(dstpath=dstpath,
               filename=filename,
               output=output,
               images_paths=image_names,
               sep=sep)
    

def save_infer(dstpath: str,
               filename: str,
               output: list[list[tuple[int, str, float]]],
               images_paths: list[str],
               sep=','
               ) -> None:
    """
    Save the inference results to a file.

    Args:
        dstpath (str): The destination path where the file will be saved.
        filename (str): The name of the file.
        output (list[list[tuple[int, str, float]]]): The inference results.
        images_paths (list[str]): The paths of the input images.
        sep (str, optional): The separator used in the file. Defaults to ','.
    """
    file = os.path.join(dstpath, filename)
    k = len(output[0])

    header = f'Image{sep}'
    for j in range(k):
        header += f'Prediction {j + 1}{sep}Confidence {j + 1} (en %){sep}'

    with open(file, 'w', encoding='utf8') as f:
        f.write(header[:-len(sep)] + '\n')

        for i in range(len(output)):
            line = f'{images_paths[i]}{sep}'
            for j in range(k):
                line += f'{output[i][j][1]}{sep}{output[i][j][2] * 100:.0f}{sep}'
            f.write(line[:-len(sep)] + '\n')
        f.close()

    print(f'Inference results saved at {file}')



if __name__ == '__main__':
    import yaml

    logging_path = os.path.join('logs', 'resnet_img256_0')
    datapath = os.path.join('data', 'images_to_predict')
    config = EasyDict(yaml.safe_load(open(os.path.join(logging_path, 'config.yaml'))))
    
    infer(infer_images_path=None,
          infer_datapath=datapath,
          logging_path=logging_path,
          plot_saliency=True,
          config=config,
          run_temperature_optimization=False,
          dstpath='',
          filename='inference_results.csv')
        