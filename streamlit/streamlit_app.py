# Python
import os
import cv2
import sys
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy.special import softmax
import torch
from torchvision import transforms
from os.path import dirname as up

#custom imports
sys.path.append(up(up(os.path.abspath(__file__))))
from src.dataloader.labels import LABELS
from utils import utils
from config.utils import load_config
from src.model import finetune_resnet

# Set page config
st.set_page_config(page_title="Blood Stain Classification App", layout="centered")

# Sidebar title
st.sidebar.title("Parameters")

# Sidebar options
save_results = st.sidebar.checkbox('Save Results', value=False)
global_path = st.sidebar.text_input('Global Path')
class_names = LABELS
plot_saliency = st.sidebar.checkbox('Plot Saliency Map', value=True)

# Function to find .yaml files in a directory
def find_yaml_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.yaml'):
                yield os.path.join(root, file)

# Config selection
config_options = list(find_yaml_files('logs'))
config_file = st.selectbox("Select Config", config_options)

# Image upload
image_files = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

# Initialize session state
if "image_files" not in st.session_state:
    st.session_state["image_files"] = None

if "image_files" is not None:
    st.session_state["image_files"] = image_files

# Display uploaded images
if st.session_state["image_files"] is not None:
    for i, image_file in enumerate(image_files):
        image = Image.open(image_file)
        st.image(image, caption=f'Uploaded Image {i+1}.',width=500)

# Temperature parameter
temperature = 1.5

# Inference
if st.button("Inference"):
    if st.session_state["image_files"] is not None:
        config = load_config(config_file)
        device = utils.get_device(device_config="cuda:0")
        config_in_log_dir = os.path.dirname(config_file)
        model = finetune_resnet.get_finetuneresnet(config)
        weight = utils.load_weights(config_in_log_dir, device=device)
        model.load_dict_learnable_parameters(state_dict=weight, strict=True)
        model = model.to(device)
        transform =  transforms.Compose([transforms.ToTensor(),transforms.Resize((256, 256))])
        results_df = pd.DataFrame(columns=['Image Name', 'Predicted Label'])

        for i,image in enumerate(image_files):
            st.write(f"Inferring on image {i+1}...")
            st.write("##############################################")
            st.write("Plotting the probability distribution of the classes...")
            model = model.to(device)
            with st.spinner('Inferring...'):
                image = Image.open(image)
                image_np = np.array(image)
                image_resized_np = cv2.resize(image_np, (256, 256))
                image = transform(image_np).unsqueeze(0).to(device)
                model.eval()
                with torch.no_grad():
                    y_pred = model.forward(image)
                numpy_y_pred_prob = y_pred.cpu().numpy()[0]
                numpy_y_pred_prob = softmax(numpy_y_pred_prob/temperature)
                numpy_y_pred_prob = numpy_y_pred_prob.reshape(1, -1)
                y_pred_df = pd.DataFrame(numpy_y_pred_prob, columns=class_names)
                y_pred_df = y_pred_df.T
                st.bar_chart(y_pred_df)
                res=class_names[y_pred.argmax(dim=-1).item()]
                st.success(f"**This image is classified as:** {res}")
                results_df = results_df.append({'Image Name': image_files[i].name, 'Predicted Label': res}, ignore_index=True)

                if save_results:
                    path = global_path if global_path else os.getcwd()
                    os.makedirs(path, exist_ok=True)
                    results_df.to_csv(os.path.join(path, 'results.csv'), index=False)
                    saliency_maps_path = os.path.join(path, 'saliency_maps')
                    os.makedirs(saliency_maps_path, exist_ok=True)

                if plot_saliency:
                    image = image.cpu()
                    model = model.cpu()
                    from src.grad_cam import get_saliency_map, threshold_and_find_contour
                    saliency_map, _ = get_saliency_map(model, image, return_label=True)
                    segmented_image = threshold_and_find_contour(saliency_map, threshold_value=125)
                    col1, col2 = st.columns(2)
                    col1.image(saliency_map, caption='Saliency Map', use_column_width=True)
                    col2.image(image_resized_np, caption='Original Image', use_column_width=True)
                    if save_results:
                        cv2.imwrite(os.path.join(saliency_maps_path, f'saliency_map_{image_files[i].name.split(".")[0]}.png'), saliency_map)

        st.dataframe(results_df)

if __name__ == '__main__':
    pass