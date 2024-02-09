import os
import random
import numpy as np
from numpy import ndarray

from PIL import Image
from typing import Tuple
import matplotlib.pyplot as plt


def segment_image_file(image: ndarray) -> ndarray:
    """
    Open the image and segment the blood stain in the image using the red colour
    """
    if image.max() < 10:
        print("Attention, l'image doit être codé en int")

    masked_image = np.zeros_like(image, dtype=int)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            masked_image[i, j] = mask_red_pixel(image[i, j])

    return masked_image


# def mask_red_pixel(pixel: Tuple[int, int, int],
#                    seuil: Tuple[int, int, int]=(58, 50, 50)
#                    ) -> bool:
#     red = (pixel[0] > seuil[0])
#     green = (pixel[1] > seuil[1])
#     blue = (pixel[2] > seuil[2])
#     return red and not(green) and not(blue)

def mask_red_pixel(pixel: Tuple[int, int, int]) -> bool:
    r, g, b = pixel
    if max(pixel) != r:
        return False

    red_seuil = (r > 70)
    green_seuil = (g < 100)
    blue_seuil = (b < 100)

    dif_r_mean_gb = r - (g + b) / 2
    test_diff = dif_r_mean_gb > 70
    # r > 20 + bg/2 

    tests = (red_seuil, green_seuil, blue_seuil, test_diff)
    nb_true = 0
    for test in tests:
        if test:
            nb_true += 1
    
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
    test_path = os.path.join('data', 'data_labo', 'test_128')
    image = get_random_img(test_path)
    mask = segment_image_file(image=image)
    plot_img_and_mask(image, mask)