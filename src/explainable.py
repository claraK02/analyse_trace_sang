import torch
from segment_anything import sam_model_registry
import cv2
from segment_anything import SamAutomaticMaskGenerator
from segment_anything import SamPredictor, sam_model_registry
from PIL import Image
import requests
from transformers import SamModel, SamProcessor
import torch
import cv2
import numpy as np




def launch():
    #torch.backends.cudnn.enabled = False
    torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    #print the shape of the image
    print(raw_image.size)
    #reduce the size of the image to 128x128
    raw_image = raw_image.resize((64,64))
    input_points = [[[12, 12]]]  # 2D location of a window in the image

    inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    )
    return masks



def generate_random_mask(size, num_ellipses, diversity=1.0):
    mask = np.zeros((size, size), dtype=np.uint8)

    for _ in range(num_ellipses):
        # Generate random center, size and angle for the ellipse
        center = tuple(np.random.normal(loc=size // 2, scale=(size // 4) * diversity, size=2).astype(int))
        axes = tuple(np.random.normal(loc=size // 6, scale=(size // 8) * diversity, size=2).astype(int))
        angle = np.random.uniform(0, 360)

        # Ensure the generated parameters are within the valid range
        center = tuple(max(min(c, size - 1), 0) for c in center)
        axes = tuple(max(min(a, size - 1), 0) for a in axes)

        # Draw the ellipse on the mask
        cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)

    return mask


def calculate_ovality(mask):
    """
    Calculates the ovality of a given mask.

    Parameters:
    - mask: A binary image mask.

    Returns:
    - ovality: The ovality value, which is the ratio of the maximum and minimum ellipse diameters.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_stain = max(contours, key=cv2.contourArea)
    ellipse = cv2.fitEllipse(main_stain)
    ovality = max(ellipse[1]) / min(ellipse[1])
    return ovality

def count_satellites(mask):
    """
    Counts the number of satellite stains in the given mask.

    Parameters:
    mask (numpy.ndarray): Binary mask representing the stained regions.

    Returns:
    int: Number of satellite stains.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    satellites = num_labels - 1  # Subtract 1 to exclude the main stain itself
    return satellites

def calculate_irregularity(mask):
    """
    Calculates the irregularity of a given mask.

    Parameters:
    - mask: A binary image mask.

    Returns:
    - irregularity: The irregularity value of the mask.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_stain = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(main_stain, True)
    area = cv2.contourArea(main_stain)
    perfect_circle_perimeter = 2 * np.pi * (area / np.pi)**0.5
    irregularity = perimeter - perfect_circle_perimeter
    return irregularity

def calculate_satellite_ratio(mask):
    """
    Calculates the average ratio of the size of satellite stains to the size of the main stain.

    Parameters:
    mask (numpy.ndarray): Binary mask representing the stained regions.

    Returns:
    float: Average ratio of the size of satellite stains to the size of the main stain.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels < 2:  # If there are no satellites, return 0
        return 0

    # The first label is the background, so the main stain is the second label
    main_stain_size = stats[1, cv2.CC_STAT_AREA]

    # The rest of the labels are the satellites
    satellite_sizes = stats[2:, cv2.CC_STAT_AREA]

    # Calculate the ratio for each satellite and return the average
    ratios = satellite_sizes / main_stain_size


    return ratios.mean()


if __name__ == "__main__":
    
    launch()
    #generate a random mask
    # mask = generate_random_mask(512, 10,diversity=1)

    # #plot the mask
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 10))
    # plt.imshow(mask, cmap='gray')
    # plt.show()

    # #calculate the ovality
    # ovality = calculate_ovality(mask)
    # print("ovality:",ovality)
     
    # #calculate the number of satellites
    # satellites = count_satellites(mask)
    # print("satellites:",satellites)
    
    # #calculate the irregularity
    # irregularity = calculate_irregularity(mask)
    # print("irregularity:",irregularity)

    # #calculate the satellite ratio
    # satellite_ratio = calculate_satellite_ratio(mask)
    # print("satellite_ratio:",satellite_ratio)
