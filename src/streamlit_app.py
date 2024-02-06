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


# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


st.title("Image Classification App")

# Path to the training data
train_data_path = 'data/data_retouche/train_128'

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
        weight = utils.load_weights(config_in_log_dir, device=device)
        print("config_in_log: ", config_in_log_dir)
        model = resnet.get_resnet(config)
        weight = utils.load_weights(config_in_log_dir, device=device)
        model.load_dict_learnable_parameters(state_dict=weight, strict=True)
        model = model.to(device)
        del weight

        # Prepare image
        image = transform(image).unsqueeze(0).to(device)

        # Inference
        model.eval()
        with torch.no_grad():
            y_pred = model.forward(image)

        st.balloons()

        # Display result
        res=class_names[y_pred.argmax(dim=1).item()]
        st.success(f"**This image is classified as:** {res}")

