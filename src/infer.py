import os
import sys
from tqdm import tqdm
from typing import List
from easydict import EasyDict
from os.path import dirname as up

import torch
from torch import Tensor
from torch.utils.data import DataLoader

sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(up(up(os.path.abspath(__file__))))

from src.dataloader.dataloader import LABELS
from src.dataloader.infer_dataloader import InferDataGenerator
from src.model import resnet
from utils import utils


def infer(datapath: str,
          logging_path: str,
          config: EasyDict,
          test_inference: bool=False
          ) -> None:
    
    device = utils.get_device(device_config=config.learning.device)

    generator = InferDataGenerator(datapath, image_size=config.data.image_size)
    infer_generator = DataLoader(dataset=generator,
                                 batch_size=config.learning.batch_size,
                                 shuffle=False,
                                 drop_last=False,
                                 num_workers=config.learning.num_workers)

    # Get model
    model = resnet.get_resnet(config)
    weight = utils.load_weights(logging_path, device=device)
    model.load_dict_learnable_parameters(state_dict=weight, strict=True)
    model = model.to(device)
    del weight

    if test_inference:
        acc: int = 0

    model.eval()
    with torch.no_grad():
        for x, images_path in infer_generator:
            x: Tensor = x.to(device)
            y_pred = model.forward(x)
            y_pred_argmax = torch.argmax(y_pred, dim=-1)
            label_prediction = list(map(lambda yi: LABELS[yi], y_pred_argmax))
            for i, image_path in enumerate(images_path):
                # print(f'image_path: {image_path} and prediction: {label_prediction[i]}')
                if test_inference:
                    result: bool = label_prediction[i] in image_path
                    print(f'prediction: {result}')
                    acc += int(result)
    
    if test_inference:
        print(f'accuracy: {100 * acc / len(generator)}%')



if __name__ == '__main__':
    import yaml

    logging_path = os.path.join('logs', 'resnet_0')
    # datapath = r'data\data_labo\test_512\1- Modèle Traces passives\bois\257.jpg'
    datapath = r'data\data_labo\test_512'
    config = EasyDict(yaml.safe_load(open(os.path.join(logging_path, 'config.yaml'))))
    
    infer(datapath, logging_path, config, test_inference=True)
        