import os
import sys
import yaml
import numpy as np
from easydict import EasyDict
import matplotlib.pyplot as plt
from os.path import dirname as up

import torch
from torch import Tensor
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

sys.path.append(up(up(os.path.abspath(__file__))))

from src.model.finetune_resnet import FineTuneResNet
from src.model.resnet import get_original_resnet
from src.model.finetune_resnet import get_finetuneresnet
from utils import utils



def get_saliency_map(model: FineTuneResNet,
                     image: np.ndarray | Tensor,
                     return_label: bool = False
                     ) -> np.ndarray | tuple[np.ndarray, Tensor]:
    """
    Generates a saliency map for the given image using the provided model.

    Args:
        model: (torch.nn.Module): The model used for generating the saliency map.
        image: (numpy.ndarray or torch.Tensor): The input image for which the saliency map is generated.
            If `image` is a numpy array, it should have shape (height, width, channels).
            If `image` is a torch tensor, it should have shape (batch_size, channels, height, width).
        plot_map (bool, optional): Whether to plot and display the saliency map. Default is False.
        return_label (bool, optional): Whether to return the output label along with the saliency map. Default is False.

    Returns:
        numpy.ndarray: The saliency map as a numpy array with shape (height, width, channels).
        If `return_label` is True, the output label is also returned as a torch.Tensor.

    """
    output = model.forward(image)
    print(f"prédiction du resnet: {torch.argmax(output[0]).item()}")

    true_resnet = get_original_resnet(model) # les parametres sont tous apprenable

    print("true_resnet:", true_resnet)

    target_layer = true_resnet.layer4[1].conv2 #c'était layer4[1].conv2 avant
    print(target_layer)

    cam = GradCAM(model=true_resnet, target_layers=[target_layer])

    #aug_smooth=True, eigen_smooth=True
    grayscale_cam = cam(input_tensor=image, targets=None, aug_smooth=True, eigen_smooth=True)

    grayscale_cam = grayscale_cam[0, :]

    # Invert the colors
    #grayscale_cam = grayscale_cam.max() - grayscale_cam


    rgb_img = utils.convert_tensor_to_rgb(image=image[0], normelize=True)

    visualization = show_cam_on_image(rgb_img, grayscale_cam)

    if return_label:
        return visualization, output
    else:
        return visualization
    
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

import cv2


def threshold_and_find_contour(heatmap: np.ndarray, threshold_value: int = 125) -> np.ndarray:
    """
    Performs thresholding and finds contours on the given heatmap.

    Args:
        heatmap: (numpy.ndarray): The heatmap for which the segmentation is performed.
        threshold_value: (float, optional): The threshold value to use. Default is 0.5.

    Returns:
        numpy.ndarray: The segmented image.
    """
    # Ensure heatmap is 2D
    if len(heatmap.shape) == 3 and heatmap.shape[2] == 3:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)

    #print max and min value of the heatmap
    print("max value of heatmap: ", np.max(heatmap))
    print("min value of heatmap: ", np.min(heatmap))

    # Apply threshold
    _, thresholded = cv2.threshold(heatmap, threshold_value, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresholded.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty image to draw the contours
    contour_image = np.zeros_like(heatmap)

    # Draw the contours
    cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 1)

    return contour_image

def get_bounding_box_and_plot(image: np.ndarray, contour_image: np.ndarray) -> tuple:
    """
    Finds the largest contour from the segmented image, computes its bounding box, and returns the coordinates.
    It also plots the bounding box on the original image.

    Args:
        image: (numpy.ndarray): The original image.
        contour_image: (numpy.ndarray): The segmented image with contours.

    Returns:
        tuple: The coordinates of the bounding box in the format (x, y, width, height).
    """
    # Find contours
    contours, _ = cv2.findContours(contour_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    max_contour = max(contours, key=cv2.contourArea)

    # Compute the bounding box around the largest contour
    x, y, w, h = cv2.boundingRect(max_contour)

    # Draw the bounding box on the original image
    image_with_bbox = cv2.rectangle(image.copy(), (x, y), (x+w, y+h), (0, 255, 0), 2)

    return (x, y, w, h), image_with_bbox

def plot_saliency_maps(model, k: int):
    """
    Plots k saliency maps for random images.

    Args:
        model: The model used for generating the saliency maps.
        k: The number of saliency maps to plot.
    """
    rows = k // 3 + (k % 3 > 0)
    fig, axs = plt.subplots(rows, 3, figsize=(20, 20))
    fig.suptitle('Saliency Maps for layer4[1].conv2')

    for i in range(k):
        x, _ = utils.get_random_img(image_type='torch')
        x: Tensor = x.unsqueeze(dim=0)
        saliency_map = get_saliency_map(model, image=x)
        axs[i // 3, i % 3].imshow(saliency_map)
        axs[i // 3, i % 3].axis('off')
    
    
    plt.show()

if __name__ == '__main__':
    #get a random image
    from torchvision import transforms
    x, _ = utils.get_random_img(image_type='numpy')

    print("shape of image used",np.shape(x) )
    #we transform the image to a tensor and unsqueeze it
    #transform = transforms.Compose([transforms.ToTensor()])
    x = transforms.ToTensor()(x).unsqueeze(0)

    config_path = os.path.join('logs', 'resnet_allw_img256_1') 
    config = EasyDict(yaml.safe_load(open(os.path.join(config_path, 'config.yaml'))))

    model = get_finetuneresnet(config)
    weight = utils.load_weights(config_path, device=torch.device('cpu'))
    model.load_dict_learnable_parameters(state_dict=weight, strict=True)
    del weight

    saliency_map = get_saliency_map(model, image=x)
    segmented_image = threshold_and_find_contour(saliency_map, threshold_value=125)

    bbox_coords, image_with_bbox = get_bounding_box_and_plot(x.squeeze().numpy().transpose(1, 2, 0), segmented_image)

    print("Bounding box coordinates: ", bbox_coords)

    # Plot saliency map, region growing segmentation, and image with bounding box
    _, axes = plt.subplots(1, 3)
    axes[0].imshow(saliency_map)
    axes[0].set_title('Saliency map')
    axes[1].imshow(segmented_image)
    axes[1].set_title('Threshold and find contour')
    axes[2].imshow(image_with_bbox)
    axes[2].set_title('Image with bounding box')
    plt.show()