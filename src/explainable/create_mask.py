import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from transformers import OwlViTProcessor, OwlViTForObjectDetection



def segment_image(image_path):
    """
    Open the image and segment the blood stain in the image using the red colour
    """
    if image_path!=None:
        img = Image.open(image_path)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #make a segmentation mask using a threshold of red colour in percent of the maximum red value to find the blood stain
    threshold = 0.45
    mask = (img[:,:,0] > threshold * img[:,:,0].max()).astype(np.uint8) * 255

    #plot the original image and the mask
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.show()

    return mask


def detect_and_plot_objects(image_np):
    """
    Detects and plots objects in an image using the OwlViT model.

    Args:
        image_np (numpy.ndarray): The input image as a NumPy array.

    Returns:
        list: A list of tuples containing the detected objects. Each tuple contains the label, confidence score, and bounding box coordinates.

    """
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

    image = Image.fromarray(image_np).convert("RGB")
    texts = [["ruler" ,"ketchup","blood"]]
    inputs = processor(text=texts, images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs=outputs, threshold=0.01, target_sizes=target_sizes)

    i = 0
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

    detected_objects = []
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        detected_objects.append((text[label], round(score.item(), 3), box))
        print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")

    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor="r", facecolor="none")
        ax.add_patch(rect)
        ax.text(box[0], box[1], f"{text[label]} {round(score.item(), 3)}", bbox=dict(facecolor="white", alpha=0.5))
    plt.axis("off")
    plt.show()

    return detected_objects


def segment_rectangular_objects(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 30, 250)

    # Perform a dilation and erosion to close gaps in between object edges
    edges = cv2.dilate(edges, None, iterations=2)
    edges = cv2.erode(edges, None, iterations=1)

    # Find contours in the edge map
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []

    # Loop over the contours
    for contour in contours:
        # Approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # If the approximated contour has four points, then it is a rectangle
        if len(approx) == 4:
            rects.append(approx)

    return rects



def replace_black_white_pixels(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find black and white pixels
    black_pixels = np.where(gray < 75)
    white_pixels = np.where(gray > 235)

    # Calculate the mean pixel value of original image
    mean_pixel_value = image.mean()

    # Replace black and white pixels with the mean pixel value
    image[black_pixels] = mean_pixel_value
    image[white_pixels] = mean_pixel_value

    
    return image



def detect_and_plot_template(image, template_paths, threshold=0.5):
    # Convert the image to grayscale
    image = np.array(image)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for template_path in template_paths:
        # Load and convert the template to grayscale
        template = cv2.imread(template_path)
       
        if template is None:
            print(f"Invalid template path: {template_path}")
            continue
        #template = np.array(template)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        w, h = template_gray.shape[::-1]

        # Check if the template image is larger than the source image
        # Check if the template image is larger than the source image
        if image_gray.shape[0] < h or image_gray.shape[1] < w:
            print(f"Template image {template_path} is larger than the source image. Resizing...")
            scale = min(image_gray.shape[0]/h, image_gray.shape[1]/w)
            template_gray = cv2.resize(template_gray, None, fx=scale, fy=scale, interpolation = cv2.INTER_AREA)
            w, h = template_gray.shape[::-1]  

        # Perform template matching
        res = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)
  
        loc = np.where(res >= threshold)

        for pt in zip(*loc[::-1]):
            # Draw a red rectangle around the matched template
            cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
            # Draw a red cross at the center of the matched template
            cv2.drawMarker(image, (pt[0] + w//2, pt[1] + h//2), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_8)

    # Display the image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

    return loc


def segment_image_file(image):
    """
    Open the image and segment the blood stain in the image using the red colour
    """

    #convert the image to numpy array
    img = np.array(image)
    #convert the image to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = replace_black_white_pixels(img)
    #print("max pixel value:",img[:,:,0].max())
    #print("min pixel value:",img[:,:,0].min())

    #make a segmentation mask using a threshold of red colour in percent of the maximum red value to find the blood stain
    threshold_red = 0.45
    threshold_green_blue = 0.7
    threshold_global = 0.1
    mask = (img[:,:,0] > threshold_red * img[:,:,0].max()) # La composante rouge est importante
   
    return mask