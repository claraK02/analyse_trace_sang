# Python
import os
import cv2
import sys
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from os.path import dirname as up

import torch
from torch import Tensor
from torchvision import transforms
from torch.nn.functional import softmax

sys.path.append(up(up(os.path.abspath(__file__))))

from utils import utils
from src.dataloader.labels import LABELS
from config.utils import load_config
from src.model import finetune_resnet
from src.gradcam import GradCam

# Set page config
st.set_page_config(page_title="Blood Stain Classification App", layout="centered")

# Sidebar title
st.sidebar.title("Parameters")

# Sidebar options
save_results = st.sidebar.checkbox('Save Results', value=False)
global_path = st.sidebar.text_input('Global Path')
plot_saliency = st.sidebar.checkbox('Plot Saliency Map', value=True)

# Image upload
image_files = st.file_uploader("Upload Image",
                               type=['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG'],
                               accept_multiple_files=True)

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
config_file = os.path.join('logs', 'retrain_resnet_allw_img256_2', 'config.yaml')

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
        model.eval()
        transform =  transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])
        results_df = pd.DataFrame(columns=['Image Name', 'Predicted Label'])

        if plot_saliency:
            gradcam = GradCam(model=model)

        for i,image in enumerate(image_files):
            st.write(f"Inferring on image {i+1}...")
            st.write("##############################################")
            st.write("Plotting the probability distribution of the classes...")
            model = model.to(device)

            with st.spinner('Inferring...'):
                image = Image.open(image)
                image: Tensor = transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    y_pred = model.forward(image)
                y_pred = softmax(y_pred / temperature, dim=-1)
                numpy_y_pred_prob = y_pred.cpu().numpy()[0].reshape(1, -1)
                y_pred_df = pd.DataFrame(numpy_y_pred_prob, columns=LABELS)
                y_pred_df = y_pred_df.T
                st.bar_chart(y_pred_df)
                res = LABELS[y_pred.argmax(dim=-1).item()]
                st.success(f"**This image is classified as:** {res}")
                results_df.loc[len(results_df)] = {'Image Name': image_files[i].name, 'Predicted Label': res}

                if save_results:
                    path = global_path if global_path else os.getcwd()
                    os.makedirs(path, exist_ok=True)
                    results_df.to_csv(os.path.join(path, 'results.csv'), index=False)
                    saliency_maps_path = os.path.join(path, 'saliency_maps')
                    os.makedirs(saliency_maps_path, exist_ok=True)

                if plot_saliency:
                    visualizations = gradcam.forward(image)
                    saliency_map: np.ndarray = visualizations[0]
                    saliency_map = saliency_map / 255
                    np_image: np.ndarray = image.cpu().numpy().squeeze().transpose(1, 2, 0)
                    col1, col2 = st.columns(2)
                    col1.image(saliency_map, caption='Saliency Map', use_column_width=True)
                    col2.image(np_image, caption='Original Image', use_column_width=True)
                    if save_results:
                        cv2.imwrite(os.path.join(saliency_maps_path, f'saliency_map_{image_files[i].name.split(".")[0]}.png'), saliency_map)

        st.dataframe(results_df)

if __name__ == '__main__':
    pass