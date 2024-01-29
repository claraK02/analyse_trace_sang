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
from easydict import EasyDict
import yaml
from dataloader.dataloader import create_dataloader
import xgboost as xgb
import os
import shap
import matplotlib.pyplot as plt

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

def segment_image_file(image):
    """
    Open the image and segment the blood stain in the image using the red colour
    """

    #convert the image to numpy array
    img = np.array(image)
    #convert the image to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #make a segmentation mask using a threshold of red colour in percent of the maximum red value to find the blood stain
    threshold = 0.45
    mask = (img[:,:,0] > threshold * img[:,:,0].max()).astype(np.uint8) * 255

    # #plot the original image and the mask
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 10))
    # plt.subplot(1, 2, 1)
    # plt.imshow(img)
    # plt.subplot(1, 2, 2)
    # plt.imshow(mask, cmap='gray')
    # plt.show()

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

            # Append the features and label to their respective lists
            features.append([ovality, satellites, irregularity])
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

            # Append the features and label to their respective lists
            test_features.append([ovality, satellites, irregularity])
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
    for i, score in enumerate(model.feature_importances_):
        print(f"Feature {i}: {score}")


    #Open the test_paths.txt file
    with open('src/test_paths.txt', 'r', encoding='utf-8') as f:
        test_paths = f.readlines()
    
    #load all the images
    images = []
    for path in test_paths:
        path = path.strip()
        print("PATH:",path)
        if os.path.isfile(path):
            image = Image.open(path)
            images.append(image)
        else:
            print(f"Invalid file path: {path}")

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

        # Append the features and label to their respective lists
        features=np.array([ovality, satellites, irregularity])

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
        axs[1].barh(range(len(shap_values[0])), shap_values[0])
        axs[1].set_yticks(range(len(shap_values[0])))
        axs[1].set_yticklabels(["Ovality", "Satellites", "Irregularity"])
        axs[1].set_title('SHAP Values')

        # PLot the segmented image on the third subplot
        axs[2].imshow(mask, cmap='gray')
        axs[2].set_title('Segmented Image')


        # Display the plots
        plt.show()
        


    
    






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
    try:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        main_stain = max(contours, key=cv2.contourArea)
        ellipse = cv2.fitEllipse(main_stain)
        ovality = max(ellipse[1]) / min(ellipse[1])
    except:
        ovality = 0
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
    
    train_xgboost()
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
    # Open the test_paths.txt file
    with open('src/test_paths.txt', 'r', encoding='utf-8') as f:
        test_paths = f.readlines()

    #create a csv file to save the attributes and class of each image
    
