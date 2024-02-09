import torch
import cv2
from segment_anything import SamAutomaticMaskGenerator
from segment_anything import SamPredictor, sam_model_registry
from PIL import Image
import requests
from transformers import SamModel, SamProcessor
import torch
import numpy as np
from easydict import EasyDict
import yaml
from dataloader.dataloader import create_dataloader
import xgboost as xgb
import os
import shap
import matplotlib.pyplot as plt
import requests
from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from scipy import ndimage

from scipy.ndimage import center_of_mass as calculate_center_of_mass

def segment_image(image_path):
    """
    Open the image and segment the blood stain in the image using the red colour
    """
    #open the image
    if image_path!=None:
        img = Image.open(image_path)
    #convert the image to numpy array
    img = np.array(img)
    #convert the image to RGB
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

from skimage import feature, io

template_path="src/template.jpeg"

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


def train_xgboost():
    '''
    train a xgboost model on the features extracted from the images
    '''
    config=EasyDict(yaml.safe_load(open('config/config.yaml')))

    #create a dataloader
    train_generator = create_dataloader(config=config, mode='train')

    #create a xgboost model
    model = xgb.XGBClassifier()

    #extract the features from the images
    features = []
    labels = []
    for i, (batch_x,batch_y,_) in enumerate(train_generator):
        print("Treating batch:",i,"out of",len(train_generator))
        for image, label in zip(batch_x, batch_y):
            # Segment the image
            mask = segment_image_file(image.permute(1,2,0).numpy())

            # Extract features
            ovality = calculate_ovality(mask)
            satellites = count_satellites(mask)
            irregularity = calculate_irregularity(mask)
            ratio=calculate_satellite_ratio(mask)

            # Append the features and label to their respective lists
            features.append([ovality, satellites, irregularity, ratio])
            labels.append(label)

    # Convert lists to numpy arrays
    features = np.array(features)
    print("FEATURES:",features)
    labels = np.array(labels)

    # Train the model
    model.fit(features, labels)

    # Create a test generator
    test_generator = create_dataloader(config=config, mode='test')

    # Evaluate the model on the test set
    test_features = []
    test_labels = []
    for i, (batch_x, batch_y,_) in enumerate(test_generator):
        for image, label in zip(batch_x, batch_y):
            # Segment the image
            mask = segment_image_file(image.permute(1,2,0).numpy())

            # Extract features
            ovality = calculate_ovality(mask)
            satellites = count_satellites(mask)
            irregularity = calculate_irregularity(mask)
            ratio=calculate_satellite_ratio(mask)

            # Append the features and label to their respective lists
            test_features.append([ovality, satellites, irregularity, ratio])
            test_labels.append(label)

    # Convert lists to numpy arrays
    test_features = np.array(test_features)
    test_labels = np.array(test_labels)

    # Calculate accuracy
    accuracy = model.score(test_features, test_labels)
    print("Test Accuracy: ", accuracy)

    # Print feature importance
    print("Feature Importance:")
    print("Feature 0: Ovality")
    print("Feature 1: Satellites")
    print("Feature 2: Irregularity")
    print("Feature 3: Satellite Ratio")
    for i, score in enumerate(model.feature_importances_):
        print(f"Feature {i}: {score}")

    #create the test dataloader
    test_generator = create_dataloader(config=config, mode='test')
    
    #load all the images
    images = []
    for i, (batch_x, batch_y,_) in enumerate(test_generator):
        for image, label in zip(batch_x, batch_y):
            images.append(image.permute(1,2,0).numpy())
    

    # Create a SHAP explainer
    explainer = shap.TreeExplainer(model)
    print("EXPLAINER created")

    for k in images:
        print("Treating image:",k)

        #reshape the image in 128*128
        k = k.resize((128,128))

        # Segment the image
        mask = segment_image_file(k)

        # Extract features
        ovality = calculate_ovality(mask)
        satellites = count_satellites(mask)
        irregularity = calculate_irregularity(mask)
        ratio=calculate_satellite_ratio(mask)

        # Append the features and label to their respective lists
        features=np.array([ovality, satellites, irregularity, ratio])

        #compute XGBoost prediction
        prediction = model.predict(np.reshape(features, (1, -1)))
        print("XGBoost prediction:",prediction)
        print("The image belong to model:",["Passive","Glissée","impact"][prediction[0]])
        predicted_class = ["Passive","Glissée","impact"][prediction[0]]
        # Get the SHAP values for the first prediction
        shap_values = explainer.shap_values(np.reshape(features, (1, -1)))
        print("SHAP values:",shap_values)

        # Plot the SHAP values with a bar plot
        # Create a new figure with 2 subplots
        fig, axs = plt.subplots(1, 3, figsize=(20, 10))
        # Add a title to the figure with the predicted class
        fig.suptitle(f'The prediction of the model is: {predicted_class}', fontsize=16)

        # Plot the image on the first subplot
        axs[0].imshow(k)
        axs[0].set_title('Processed Image')

        # Plot the SHAP values with a bar plot on the second subplot
        axs[1].bar(["Ovality", "Satellites", "Irregularity", "Satellite Ratio"], shap_values[0])
        axs[1].set_title('SHAP Values')

        # PLot the segmented image on the third subplot
        axs[2].imshow(mask, cmap='gray')
        axs[2].set_title('Segmented Image')


        # Display the plots
        plt.show()

    return model

def inference_xgboost(model, image):
    """
    input: image, a numpy array
    output: the prediction of the model
    """
    # Segment the image
    mask = segment_image_file(image)

    # Extract features
    ovality = calculate_ovality(mask)
    satellites = count_satellites(mask)
    irregularity = calculate_irregularity(mask)
    ratio=calculate_satellite_ratio(mask)

    # Append the features and label to their respective lists
    features = np.array([ovality, satellites, irregularity, ratio])

    # Compute XGBoost prediction
    prediction = model.predict(np.reshape(features, (1, -1)))
                    
    #compute the SHAP values
    # Create a SHAP explainers
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(np.reshape(features, (1, -1)))


    return prediction, shap_values

        


    
    






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
    mask = mask.astype(np.uint8)
    try:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        main_stain = max(contours, key=cv2.contourArea)
        ellipse = cv2.fitEllipse(main_stain)
        ovality = max(ellipse[1]) / min(ellipse[1])
    except:
        ovality = 0
    return ovality



def classify_distribution(segmentation_mask):
    # Convert the image to a numpy array
    img = np.array(segmentation_mask)

    # If the image is not binary, convert it to grayscale and threshold it
    if np.unique(img).size > 2:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, blood_pixels = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    else:
        blood_pixels = img

    # Convert the binary mask to boolean
    blood_pixels = blood_pixels.astype(bool)

    # Calculate the center of mass of the blood stain
    center_of_mass = calculate_center_of_mass(blood_pixels)

    # Calculate the distance of each blood pixel to the center of mass
    y_indices, x_indices = np.indices(blood_pixels.shape)
    distances = np.sqrt((x_indices - center_of_mass[0])**2 + (y_indices - center_of_mass[1])**2)
    
    # Calculate the mean and standard deviation of the distances
    mean_distance = distances[blood_pixels].mean()
    std_distance = distances[blood_pixels].std()

    # Classify the distribution based on the mean and standard deviation
    if std_distance < mean_distance * 0.5:
        return 0  # Central distribution
    elif std_distance < mean_distance:
        return 1  # Linear distribution
    elif std_distance < mean_distance * 1.5:
        return 2  # Curvilinear distribution
    else:
        return 3  # Burst distribution
    

import numpy as np

def calculate_homogeneity(mask):
    """
    Calculates the homogeneity of the distribution of millimetric traces around the centimetric trace.

    Parameters:
    mask (numpy.ndarray): Binary mask representing the stained regions.

    Returns:
    float: Homogeneity of the distribution of millimetric traces.
    """
    mask = mask.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels < 3:  # If there are no satellites, return 0
        return 0

    # The first label is the background, so the main stain is the second label
    main_stain_centroid = centroids[1]

    # The rest of the labels are the satellites
    satellite_centroids = centroids[2:]

    # Calculate the distance from each satellite to the main stain
    distances = np.sqrt(np.sum((satellite_centroids - main_stain_centroid)**2, axis=1))

    # Calculate the variance of the distances
    variance = np.var(distances)

    # A lower variance means a more homogeneous distribution
    homogeneity = 1 / variance if variance != 0 else 0

    return homogeneity

def count_internal_striations(mask):
    """
    Counts the number of internal striations in the given mask.

    Parameters:
    mask (numpy.ndarray): Binary mask representing the stained regions.

    Returns:
    int: Number of internal striations.
    """
    try:
        mask = mask.astype(np.uint8)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        # The hierarchy returned by cv2.findContours has the following form:
        # [Next, Previous, First_Child, Parent]
        # If the Parent is not -1, then the contour is internal
        internal_striations = sum(1 for contour, info in zip(contours, hierarchy[0]) if info[3] != -1)
        
        return internal_striations
    except:
        return 0

def count_satellites(mask):
    """
    Counts the number of satellite stains in the given mask.

    Parameters:
    mask (numpy.ndarray): Binary mask representing the stained regions.

    Returns:
    int: Number of satellite stains.
    """
    mask = mask.astype(np.uint8)
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
    mask = mask.astype(np.uint8)
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

    mask = mask.astype(np.uint8)
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
    
    template_path="src/template.jpeg"
    list_template_paths=["src/template.jpeg","src/template_2.jpeg","src/template_3.jpeg","src/template_4.jpeg"]
    model=train_xgboost() #get the trained model

    with open('test_paths.txt', 'r', encoding='utf-8') as f:
        test_paths = f.readlines()

    #make an inference
    #open the first image
        
    path=test_paths[0].strip()

    if os.path.isfile(path):
        image = Image.open(path)
        image = np.array(image)
        prediction, shap_values = inference_xgboost(model, image)
        print("Prediction:",prediction)
        print("SHAP values:",shap_values)





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

    #open one path from the test_paths.txt file
    #Open the test_paths.txt file
    with open('test_paths.txt', 'r', encoding='utf-8') as f:
        test_paths = f.readlines()

    #obtain the segmentation mask of the image
    #open image
    for k in range(len(test_paths)):
        image = Image.open(test_paths[k].strip())
        mask = segment_image_file(image)
        #calculate the number of internal striations
        internal_striations = count_internal_striations(mask)
        #calculate the homogeneity
        homogeneity = calculate_homogeneity(mask)
        print("homogeneity:",homogeneity)
        print("internal_striations:",internal_striations)
        # object=detect_and_plot_objects(np.array(image))
        # #plot the mask and print the distribution associated
        # import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        plt.imshow(mask, cmap='gray')
        plt.show()
    # print("Distribution:",classify_distribution(mask))

    #tableau de correspondance: 0: central, 1: linear, 2: curvilinear, 3: burst

    #On parcours les images du dossier et on détecte les régions rectangulaires
    
    # for path in test_paths:
    #     print("Treating image:",path)
    #     path = path.strip()
    #     if os.path.isfile(path):
    #         image = Image.open(path)
    #         coord=detect_and_plot_template(image, list_template_paths, threshold=0.4)
    #         print("COORD:",coord)
    #         #segment_image_file(image)
    #     else:
    #         print(f"Invalid file path: {path}")
    


    # #open the first image
    # image = Image.open(test_paths[0].strip())
    # #image=image.resize((128,128))

    # #convert the image to numpy array
    # image = np.array(image)

    # #segment rectangular objects
    # rects = segment_rectangular_objects(image)
    # print("rects:",rects)

    #     # Draw each rectangle on the image
    # for rect in rects:
    #     cv2.polylines(image, [rect], True, (0, 255, 0), 2)

    # # Display the image
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.show()

    # #get the mask and plot image and mask side by side
    # # mask = segment_image_file(image)
    # # import matplotlib.pyplot as plt
    # # plt.figure(figsize=(10, 10))
    # # plt.subplot(1, 2, 1)
    # # plt.imshow(image)
    # # plt.subplot(1, 2, 2)
    # # plt.imshow(mask, cmap='gray')
    # # plt.show()
    
