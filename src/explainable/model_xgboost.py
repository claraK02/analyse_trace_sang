import os
import shap
import yaml
import numpy as np
from PIL import Image
import xgboost as xgb
from easydict import EasyDict
import matplotlib.pyplot as plt


from src.dataloader.dataloader import create_dataloader
from src.explainable.create_mask import segment_image_file
from src.explainable.find_criteres import calculate_ovality, count_satellites, \
                                          calculate_irregularity, calculate_satellite_ratio


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


    #Open the test_paths.txt file
    with open('test_paths.txt', 'r', encoding='utf-8') as f:
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


