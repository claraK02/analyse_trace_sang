import numpy as np
from numpy import ndarray

from PIL import Image
from typing import Tuple
import matplotlib.pyplot as plt


def segment_image_file(image: ndarray) -> ndarray:
    """
    Open the image and segment the blood stain in the image using the red colour
    """
    # image = np.array(image)
    if image.max() < 10:
        print("Attention, l'image doit être codé en int")

    masked_image = np.zeros_like(image, dtype=int)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            masked_image[i, j] = mask_red_pixel(image[i, j])
    
    # print(masked_image)
    print(masked_image.sum())

    return masked_image


def mask_red_pixel(pixel: Tuple[int, int, int],
                   seuil: Tuple[int, int, int]=(58, 50, 50)
                   ) -> bool:
    red = (pixel[0] > seuil[0])
    green = (pixel[1] > seuil[1])
    blue = (pixel[2] > seuil[2])
    return red and not(green) and not(blue)


if __name__ == '__main__':
    image_path = r'data\data_labo\test_128\4- Modèle Transfert glissé\bois\27.jpg'
    image = np.array(Image.open(image_path))
    mask = segment_image_file(image=image)

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(image)
    axes[0].set_title('image')
    axes[1].imshow(mask * 255)
    axes[1].set_title('mask') 
    plt.show()