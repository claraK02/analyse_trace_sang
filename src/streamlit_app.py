import streamlit as st
import os
import argparse
from PIL import Image
from torchvision import transforms
from easydict import EasyDict
import torch
import sys
from torch import Tensor

#add the one level up directory to the sys path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.config import load_config, find_config
from src import train, train_adversarial, test
from config.config import test_logger
from src.dataloader.dataloader import create_dataloader
from metrics import Metrics
from src.model import resnet
from utils import utils
import numpy as np


st.title("Image Classification App")

# Path to the training data
train_data_path = 'data/data_retouche/train_512'

#make a list of all the paths of the folders in data/data_labo the path should have the shape data/data_labo/train_512, data/data_labo/val_512, data/data_labo/test_512
list_paths = [os.path.join('data/data_labo', folder) for folder in os.listdir('data/data_labo')]

#make a choice box to select the path
train_data_path = st.selectbox("Select the training data path", list_paths)

# Get the list of directories in the training data path
class_names = [name for name in os.listdir(train_data_path) if os.path.isdir(os.path.join(train_data_path, name))]

#print the class names
print(class_names)


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
if image_file is not None:
    image = Image.open(image_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

# Inference button
if st.button("Inference"):
    if image_file is not None:
        # Load config
        config = load_config(config_file)

        # Get device
        device = utils.get_device(device_config=config.learning.device)

        print("config_file: ", config_file)
        config_in_log_dir = os.path.dirname(config_file)
        print("config_in_log_dir: ", config_in_log_dir)
        model = resnet.get_resnet(config)
        weight = utils.load_weights(config_in_log_dir, device=device)
        model.load_dict_learnable_parameters(state_dict=weight, strict=True)
        model = model.to(device)
        del weight

        print("max of image: ", np.array(image).max())
        print("min of image: ", np.array(image).min())
       
        # Prepare image
        image = torch.from_numpy(np.array(image)/255).float()
        image = image.permute(2, 0, 1).unsqueeze(0).to(device)
       
        print("image.shape: ", image.shape)
        print("image", image)

        # Inference
        model.eval()
        with torch.no_grad():
            y_pred = model.forward(image)

        st.balloons()

        # Display result
        res=class_names[y_pred.argmax(dim=1).item()]
        st.success(f"**This image is classified as:** {res}")

