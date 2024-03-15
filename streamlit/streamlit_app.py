import os
import sys
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from os.path import dirname as up
from PIL import Image, ImageEnhance
import io
from scipy.special import softmax
from streamlit_image_comparison import image_comparison

import torch
from torchvision import transforms

sys.path.append(up(up(os.path.abspath(__file__))))

from src.dataloader.dataloader import LABELS
from src.explainable.create_mask import mask_red_pixel_hsv
from utils import utils
from config.utils import load_config
from src.model import finetune_resnet
import torch.nn.functional as F

# set page config
st.set_page_config(page_title="Image-Comparison Example", layout="centered")

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
image_files = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
print("IMAGE FILE: ", image_files)
# Add sliders for image transformations
rotation = st.sidebar.slider('Rotation', min_value=0, max_value=180, value=0)
hflip = st.sidebar.checkbox('Horizontal Flip', value=False)
vflip = st.sidebar.checkbox('Vertical Flip', value=False)
contrast = st.sidebar.slider('Contrast', min_value=-3.0, max_value=3.0, value=1.0)
brightness = st.sidebar.slider('Brightness', min_value=-3.0, max_value=3.0, value=1.0)

#choice to print interpretability criteria or not:
interpretability = st.checkbox('Print Interpretability Criteria', value=False)


# Initialisation de st.session_state si nécessaire
if "image_files" not in st.session_state:
    st.session_state["image_files"] = None
if "barchart_placeholder" not in st.session_state:
    st.session_state["barchart_placeholder"] = st.empty()



if image_files is not None:
    st.session_state["image_files"] = image_files

# Utilisation de st.session_state pour stocker le placeholder du graphique barchart
barchart_placeholder = st.session_state["barchart_placeholder"]

# Utilisation de st.session_state pour récupérer les images
if st.session_state["image_files"] is not None:
    for i, image_file in enumerate(image_files):
            
        image = Image.open(image_file)
        # Convertissez l'image PIL en tableau numpy pour l'utiliser avec OpenCV
        image_np = np.array(image)
       

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


        st.image(image, caption=f'Uploaded Image {i+1}.',width=500)

# Add a slider in Streamlit to control the temperature parameter
temperature = st.slider('Temperature', min_value=0.01, max_value=3.0, value=1.5)
        
if st.button("Inference"):
    if st.session_state["image_files"] is not None:
        config = load_config(config_file)
        # Get device (the first GPU if available, otherwise the CPU)
        device = utils.get_device(device_config="cuda:0")
        print("config_file: ", config_file)
        config_in_log_dir = os.path.dirname(config_file)
        print("config_in_log_dir: ", config_in_log_dir)
        model = finetune_resnet.get_finetuneresnet(config) #to get the model
        weight = utils.load_weights(config_in_log_dir, device=device)
        #move weight to device
        model.load_dict_learnable_parameters(state_dict=weight, strict=True)
        model = model.to(device)
        print("model moved to device: ", device)
        #del weight
        # Prepare image
        transform =  transforms.Compose([transforms.ToTensor(),transforms.Resize((256, 256))])
        print("image resized and transformed")

        #we also resize the numpy image in the same way
        image_resized_np = np.array(cv2.resize(image_np, (256, 256)))
        
        

        # Create an empty DataFrame to store image names and predicted labels
        results_df = pd.DataFrame(columns=['Image Name', 'Predicted Label'])

        # Loop through the uploaded images           
        for i,image in enumerate(image_files):
            #we print in the app that we are inferring on the image
            st.write(f"Inferring on image {i+1}...")
            st.write("##############################################")
            #affiche une icone de chargement
            model = model.to(device) #ATTENTION: il faut remettre le modèle sur le device !! a chaque fois qu'on le charge
            with st.spinner('Inferring...'):
                #open the image
                image = Image.open(image)
                image_np = np.array(image)
                
                #apply the transform to the image
                image = transform(image_np).unsqueeze(0).to(device)

                print("image.shape: ", image.shape)
                # Inference
                model.eval()
                with torch.no_grad():
                    y_pred = model.forward(image)

                #we print y_pred:
                print("y_pred: ", y_pred)

              
                numpy_y_pred_prob = y_pred.cpu().numpy()[0]

                # Apply the softmax function with the temperature parameter
                numpy_y_pred_prob = softmax(numpy_y_pred_prob/temperature)

                print("numpy_y_pred_prob: ", numpy_y_pred_prob)

                # Reshape the array to be 2D with one row
                numpy_y_pred_prob = numpy_y_pred_prob.reshape(1, -1)

                # Convert the output tensor to a DataFrame
                y_pred_df = pd.DataFrame(numpy_y_pred_prob, columns=class_names)

                # Transpose the DataFrame so that each class has its own row
                y_pred_df = y_pred_df.T

                # Plot the probability distribution
                # Mettez à jour le graphique barchart en utilisant le placeholder
                barchart_placeholder.bar_chart(y_pred_df)

                # Display result
                res=class_names[y_pred.argmax(dim=-1).item()]
                st.success(f"**This image is classified as:** {res}")

                # Append the image name and predicted label to the results DataFrame
                results_df = results_df.append({'Image Name': image_files[i].name, 'Predicted Label': res}, ignore_index=True)

                # Générer et afficher la carte de saillance si la case est cochée
                # Generate and display the saliency map if the checkbox is checked

                if plot_saliency:
                    # Put the image and model on cpu before using them
                    image = image.cpu()
                    model = model.cpu()
                    from src.grad_cam import get_saliency_map, threshold_and_find_contour, get_bounding_box_and_plot

                    # Generate the saliency map
                    saliency_map, _ = get_saliency_map(model, image, return_label=True)


                    # Segment the saliency map
                    segmented_image = threshold_and_find_contour(saliency_map, threshold_value=125)

                    # Create two columns
                    col1, col2 = st.columns(2)

                    # Display the saliency map and the original image with the bounding box side by side
                    col1.image(saliency_map, caption='Saliency Map', use_column_width=True)
                    col2.image(image_resized_np, caption='Original Image', use_column_width=True)

                   
                    # render image-comparison
                    # image_comparison(
                    #     img1=saliency_map,
                    #     img2=image_np,
                    # )

                if interpretability:
                    from src.explainable.clip_vit_tree import classify_image_with_vit, get_criterions
                    with st.spinner('Analysing interpretability criteria...'):
                        # Get the criterions list
                        criterions = get_criterions(image_np)
                        print("criterions: ", criterions)
                        # Display the criterions as a DataFrame
                        criterions_df = pd.DataFrame(criterions, columns=['Criterions'])
                        st.write(criterions_df)
                        #print dataframe
                        print("criterions_df: ", criterions_df)

        # Display the results DataFrame after the loop
        print("results_df: ", results_df)
        st.dataframe(results_df)
        
