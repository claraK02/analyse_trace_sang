import os
import random
import numpy as np
from numpy import ndarray

from PIL import Image
from typing import Tuple
import matplotlib.pyplot as plt
import torch

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

def segment_image_kmeans(image: np.ndarray, max_clusters: int = 10) -> list:
    # Reshape the image to be a list of RGB values
    pixels = image.reshape(-1, 3)
    
    # List to hold the variance for each number of clusters
    variances = []
    
    # Calculate variance for each number of clusters
    for n_clusters in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters,  n_init=10, random_state=0).fit(pixels)
        variances.append(kmeans.inertia_)
    
    # Plot the variance as a function of the number of clusters
    plt.plot(range(1, max_clusters + 1), variances)
    plt.show()
    
    # Determine the optimal number of clusters as the elbow point in the plot
    optimal_clusters = variances.index(min(variances)) + 1
    print(f"Optimal number of clusters: {optimal_clusters}")
    
    # Apply KMeans with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=0).fit(pixels)
    
    # Create a mask for each cluster
    masks = []
    for cluster in range(optimal_clusters):
        mask = kmeans.labels_.reshape(image.shape[:2]) == cluster
        masks.append(mask)
    
    return masks


def segment_image_file(image: np.ndarray) -> np.ndarray:
    """
    Open the image and segment the blood stain in the image using the red colour
    """
    if image.max() < 10:
        print("Attention, l'image doit être codé en int")
        image = image * 255

    masked_image = mask_red_pixel(image)
    return masked_image


def batched_segmentation(batch_tensor: torch.Tensor) -> torch.Tensor:
    """
    Takes a tensor batch of images as input and returns a tensor batch of masks.
    Input shape can be something like this:  torch.Size([32, 3, 128, 128])
    """
    # Convert the tensor to numpy array
    batch_array = batch_tensor.numpy()

    # Apply the mask_red_pixel function to each image in the batch
    masked_batch = np.array([mask_red_pixel_batched(image) for image in batch_array])

    # Convert the numpy array back to tensor
    masked_tensor = torch.from_numpy(masked_batch)

    return masked_tensor

def mask_red_pixel_batched(image: np.ndarray) -> np.ndarray:
    """
    create the red mask
    """
    # Check if the pixel values are between 0 and 1 and scale them up to 0-255 if they are
    if image.max() <= 1.0:
        image = image * 255   
    r, g, b = np.split(image, 3, axis=0)
    red_seuil = (r > 70)
    green_seuil = (g < 100)
    blue_seuil = (b < 100)

    dif_r_mean_gb = r - (g + b) / 2
    test_diff = dif_r_mean_gb > 70

    tests = np.array([red_seuil, green_seuil, blue_seuil, test_diff])
    nb_true = np.sum(tests, axis=0)

    return (nb_true >= 3).astype(int)  # Convert boolean array to int

def mask_red_pixel(image: np.ndarray) -> np.ndarray:
    r, g, b = np.split(image, 3, axis=-1)
    red_seuil = (r > 70)
    green_seuil = (g < 100)
    blue_seuil = (b < 100)

    dif_r_mean_gb = r - (g + b) / 2
    test_diff = dif_r_mean_gb > 70

    tests = np.array([red_seuil, green_seuil, blue_seuil, test_diff])
    nb_true = np.sum(tests, axis=0)

    return nb_true >= 3


def plot_img_and_mask(img: ndarray, mask: ndarray) -> None:
    mul = 255 if mask.max() < 1.01 else 1
    _, axes = plt.subplots(1, 2)
    axes[0].imshow(img)
    axes[0].set_title('image')
    axes[1].imshow(mask * mul)
    axes[1].set_title(f'mask (pixel value x{mul})') 
    plt.show()


def get_random_img(data_path: str) -> ndarray:
    label = random.choice(os.listdir(data_path))
    background = random.choice(os.listdir(os.path.join(data_path, label)))
    folder_path = os.path.join(data_path, label, background)
    images = os.listdir(folder_path)

    if len(images) == 0:
        # if there are not image, find an other one
        return get_random_img(data_path)
    
    image_path = os.path.join(folder_path, random.choice(images))
    print(f'{image_path=}')
    image = np.array(Image.open(image_path))
    return image



if __name__ == '__main__':
    test_path = os.path.join('data', 'data_labo', 'test_256')
    image = get_random_img(test_path)

    #use the kmeans algorithm to segment the image
    masks = segment_image_kmeans(image)

    # Plot the image and the masks
    plot_img_and_mask(image, masks[0])
    plot_img_and_mask(image, masks[1])
    plot_img_and_mask(image, masks[2])



    # Convert the image to a tensor, add an extra dimension to simulate a batch, and transpose to PyTorch format
    image_tensor = torch.from_numpy(image.transpose((2, 0, 1))).unsqueeze(0)

    # Apply batched segmentation
    mask_tensor = batched_segmentation(image_tensor)

    # Convert the mask tensor back to a numpy array, remove the extra dimension, and transpose back to image format
    mask = mask_tensor.numpy().squeeze(0).transpose((1, 2, 0))

    # Plot the image and the mask
    plot_img_and_mask(image, mask)