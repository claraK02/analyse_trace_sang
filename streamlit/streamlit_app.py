import os
import sys
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from os.path import dirname as up
from PIL import Image, ImageEnhance

import torch
from torchvision import transforms

sys.path.append(up(up(os.path.abspath(__file__))))

from src.dataloader.dataloader import LABELS
from src.explainable.create_mask import mask_red_pixel_hsv
from utils import utils
from config.utils import load_config
from src.model import finetune_resnet
import torch.nn.functional as F

st.title("Image Classification App")

st.sidebar.title("Image Transformations")


plot_mask = st.sidebar.checkbox('Plot Mask', value=False)
# Ajoutez cette ligne après les autres contrôles de la barre latérale
hue_range = st.sidebar.slider('Hue Range', min_value=0, max_value=180, value=(0, 180))

#make a list of all the paths of the folders in data/data_labo the path should have the shape data/data_labo/train_512, data/data_labo/val_512, data/data_labo/test_512
list_paths = [os.path.join('data/data_labo', folder) for folder in os.listdir('data/data_labo')]

#class_names = [name for name in os.listdir(train_data_path) if os.path.isdir(os.path.join(train_data_path, name))]
class_names = LABELS
#print the class names
print(class_names)

# Ajoutez cette case à cocher dans la barre latérale
plot_saliency = st.sidebar.checkbox('Plot Saliency Map', value=True)



# Recursive function to find .yaml files in a directory
def find_yaml_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.yaml'):
                yield os.path.join(root, file)

# Menu for config selection
config_options = list(find_yaml_files('logs'))
config_file = st.selectbox("Select Config", config_options)


# Image upload
image_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
print("IMAGE FILE: ", image_file)
if image_file is not None:
    image = Image.open(image_file)
    # Convertissez l'image PIL en tableau numpy pour l'utiliser avec OpenCV
    image_np = np.array(image)
    # Add sliders for image transformations
    rotation = st.sidebar.slider('Rotation', min_value=0, max_value=180, value=0)
    hflip = st.sidebar.checkbox('Horizontal Flip', value=False)
    vflip = st.sidebar.checkbox('Vertical Flip', value=False)
    contrast = st.sidebar.slider('Contrast', min_value=-3.0, max_value=3.0, value=1.0)
    brightness = st.sidebar.slider('Brightness', min_value=-3.0, max_value=3.0, value=1.0)


    #apply the transformations
    #Apply transformations to the image
    image = image.rotate(rotation)
    if hflip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if vflip:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)

    enhancer_contrast = ImageEnhance.Contrast(image)
    image = enhancer_contrast.enhance(contrast)
    enhancer_brightness = ImageEnhance.Brightness(image)
    image = enhancer_brightness.enhance(brightness)

    # Appliquez le masque si le bouton "plot mask" est coché
    if plot_mask:
        #mask = mask_red_pixel_hsv(image_np, hue_min, hue_max)
        mask = mask_red_pixel_hsv(image_np, hue_range[0], hue_range[1])
    
        # Créez une image de superposition verte avec le masque
        overlay = np.zeros_like(image_np)
        overlay[mask > 0] = (0, 255, 0)  # Vert en RGB
        
        # Superposez le masque à l'image originale
        alpha = 0.5  # Définissez l'alpha pour la transparence
        image_np = cv2.addWeighted(overlay, alpha, image_np, 1 - alpha, 0)
        
        # Convertissez l'image numpy en image PIL pour l'afficher avec Streamlit
        image = Image.fromarray(image_np)


    st.image(image, caption='Uploaded Image.',width=500)


if st.button("Inference"):
    if image_file is not None:

        #affiche une icone de chargement
        with st.spinner('Inferring...'):
            config = load_config(config_file)

            # Get device
            device = utils.get_device(device_config=config.learning.device)

            print("config_file: ", config_file)
            config_in_log_dir = os.path.dirname(config_file)
            print("config_in_log_dir: ", config_in_log_dir)
            model = finetune_resnet.get_finetuneresnet(config) #to get the model

            weight = utils.load_weights(config_in_log_dir, device=device)
            model.load_dict_learnable_parameters(state_dict=weight, strict=True)
            model = model.to(device)
            del weight

            # Prepare image
            transform = transforms.Compose([transforms.ToTensor()])
            #apply the transform to the image
            image = transform(image).unsqueeze(0).to(device)

            print("image.shape: ", image.shape)
            print("image", image)

            # Inference
            model.eval()
            with torch.no_grad():
                y_pred = model.forward(image)

            #we print y_pred:
            print("y_pred: ", y_pred)

            # Apply softmax to get probability distribution
            y_pred_prob = F.softmax(y_pred, dim=-1)

            print("y_pred_prob: ", y_pred_prob)
            
            numpy_y_pred_prob = y_pred_prob.cpu().numpy()[0]

            print("numpy_y_pred_prob: ", numpy_y_pred_prob)

            # Reshape the array to be 2D with one row
            numpy_y_pred_prob = numpy_y_pred_prob.reshape(1, -1)

            # Convert the output tensor to a DataFrame
            y_pred_df = pd.DataFrame(numpy_y_pred_prob, columns=class_names)

            # Transpose the DataFrame so that each class has its own row
            y_pred_df = y_pred_df.T

            # Plot the probability distribution
            st.bar_chart(y_pred_df)

            # Display result
            res=class_names[y_pred.argmax(dim=-1).item()]
            st.success(f"**This image is classified as:** {res}")

                    # Générer et afficher la carte de saillance si la case est cochée
            if plot_saliency:
                #put the image on cpu before using it
                image = image.cpu()
                model = model.cpu()
                from src.grad_cam import get_saliency_map
                saliency_map, _ = get_saliency_map(model, image, return_label=True)

                # Create two columns
                col1, col2 = st.columns(2)

                # Display the original image and the saliency map side by side
                col1.image(image_np, caption='Original Image', use_column_width=True)
                col2.image(saliency_map, caption='Saliency Map', use_column_width=True)
