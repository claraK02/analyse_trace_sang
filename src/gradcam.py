import os
import sys
import cv2
import numpy as np
from os.path import dirname as up

import torch
from torch import Tensor
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

sys.path.append(up(up(os.path.abspath(__file__))))

from src.model.finetune_resnet import FineTuneResNet
from src.model.resnet import get_original_resnet
from utils import utils


class GradCamPlusPlus:
    def __init__(self, model: FineTuneResNet) -> None:
        """
        Initialize the GradCAM++ object.

        Args:
            model (FineTuneResNet): The fine-tuned ResNet model.
        """
        true_resnet = get_original_resnet(model)
        target_layer = true_resnet.layer4[1].conv2
        self.cam = GradCAMPlusPlus(model=true_resnet, target_layers=[target_layer])

    def forward(self, image: Tensor) -> np.ndarray:
        """
        Applies Grad-CAM++ to the input image and returns the visualizations.

        Args:
            image (Tensor): The input image tensor.

        Returns:
            np.ndarray: The visualizations of the input image after applying Grad-CAM++.
        """
        batch_size, img_c, img_h, img_w = image.shape
        grayscale_cam = self.cam(input_tensor=image,
                                 targets=None,
                                 aug_smooth=True,
                                 eigen_smooth=True)

        visualizations = np.zeros((batch_size, img_h, img_w, img_c))
        for b in range(batch_size):
            rgb_img = utils.convert_tensor_to_rgb(image=image[b].cpu(), normelize=True)
            visualizations[b] = show_cam_on_image(rgb_img, grayscale_cam[b])

        return visualizations

    def save_saliency_maps(self,
                           visualizations: np.ndarray,
                           dstpath: str,
                           filenames: list[str]
                           ) -> None:
        """
        Save the saliency maps as images.

        Args:
            visualizations (np.ndarray): The saliency maps to be saved.
            dstpath (str): The destination path where the images will be saved.
            filenames (list[str]): The filenames for the saved images.
        """
        if not os.path.exists(dstpath):
            os.mkdir(dstpath)
            print(f"Directory {dstpath} created ")

        for i, visualization in enumerate(visualizations):
            cv2.imwrite(os.path.join(dstpath, filenames[i]), visualization)

    def get_probability_with_mask(self,
                                  model: FineTuneResNet,
                                  image: Tensor,
                                  ) -> Tensor:
        """
        Calculates the probability of each class in the given image using the saliency mask.

        Args:
            model (FineTuneResNet): The model used for prediction.
            image (Tensor): The input image.

        Returns:
            Tensor: The predicted probabilities for each class.
        """
        mask = utils.normalize_image(self.forward(image=image))
        mask = torch.tensor(mask, dtype=torch.float32).permute(0, 3, 1, 2)
        mask = mask.to(model.device)
        masked_images = (image * mask)
        with torch.no_grad():
            logits = model.forward(masked_images)
        y_pred = torch.nn.functional.softmax(logits, dim=1)
        return y_pred


if __name__ == '__main__':
    import yaml
    import torch
    from easydict import EasyDict
    from src.model.finetune_resnet import get_finetuneresnet

    batch_size = 2
    x = torch.rand(batch_size, 1, 256, 256)
    for i in range(batch_size):
        xi, _ = utils.get_random_img(image_type='torch')
        x[i] = xi

    config_path = os.path.join('logs', 'retrain_resnet_allw_img256_2')
    config = EasyDict(yaml.safe_load(open(os.path.join(config_path, 'config.yaml'))))

    model = get_finetuneresnet(config)
    weight = utils.load_weights(config_path, device=torch.device('cpu'))
    model.load_dict_learnable_parameters(state_dict=weight, strict=False)
    del weight

    gradcam_plus_plus = GradCamPlusPlus(model=model)
    visualizations = gradcam_plus_plus.forward(x)
    print("visualizations shape:", visualizations.shape)
    gradcam_plus_plus.save_saliency_maps(visualizations,
                                         'gradcam_images',
                                         [f'gradcam_{i}.png' for i in range(batch_size)])

    y_pred = gradcam_plus_plus.get_probability_with_mask(model=model, image=x)
    print("y_pred shape:", y_pred.shape)
