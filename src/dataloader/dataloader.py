from typing import Tuple, Dict
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import torch
import yaml
from easydict import EasyDict
import random
import matplotlib.pyplot as plt
import numpy as np

class ImageClassificationDataGenerator(Dataset):
    def __init__(self, config: EasyDict, mode: str, validation_percentage: float = 0.2) -> None:
        assert mode in ['train', 'val', 'test'], f"Error, expected mode is train, val, or test but found '{mode}'"
        self.mode = mode
        self.transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            # Add other transformations as needed
        ])

        # We create one list of paths per class

        # Get the list of classes! they are the folder names in data_path
        self.class_labels = sorted(os.listdir('data'))
        print("class_labels:", self.class_labels)
        print(f"Found {len(self.class_labels)} classes: {self.class_labels}")

        #We have the following correspondance between class labels and class indices:
        # 0: Carrelage NH
        # 1: Carrelage H
        # 2: Carrelage V



        # Get all the paths of all the images in all the folders or subfolders or subsubfolders or subsubsubfolders in each class
        all_image_paths = []
        all_image_labels = []
        for class_label in self.class_labels:
            print(f"Searching for images in {class_label}...")
            for root, dirs, files in os.walk(os.path.join('data', class_label)):
                print(f"Found {len(files)} images in {root}")
                for file in files:
                    if file.endswith(('.png', '.jpg', '.jpeg', '.JPG')):  # add or remove file extensions as needed
                        path = os.path.join(root, file)
                        if "Retouches" in path:  # on ne prend pas les images non retouchées
                            all_image_paths.append(path)
                            all_image_labels.append(class_label)

        # Shuffle paths to ensure randomness
        c = list(zip(all_image_paths, all_image_labels))
        random.shuffle(c)
        all_image_paths, all_image_labels = zip(*c)

        # Split data into train and validation sets
        validation_size = int(validation_percentage * len(all_image_paths))
        if self.mode == 'train':
            self.all_image_paths = all_image_paths[validation_size:]
            self.all_image_labels = all_image_labels[validation_size:]
        elif self.mode == 'val':
            self.all_image_paths = all_image_paths[:validation_size]
            self.all_image_labels = all_image_labels[:validation_size]
        else:
            self.all_image_paths = all_image_paths
            self.all_image_labels = all_image_labels

        print("Il y a en tout", len(self.all_image_paths), "images dans le générateur ", self.mode)
        print(f"The {self.mode} generator was created")
        

    def __len__(self) -> int:
        return len(self.all_image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        input: index of the image to load
        output: tuple (x, y) where x is the image and y is the class label
        '''

        # Get image path and class label
        img_path = self.all_image_paths[index]
        class_label = self.all_image_labels[index]

        # Load image and apply transformations
        img = Image.open(img_path).convert('RGB')
        x = self.transform(img)

        # Convert class label to tensor
        y = torch.tensor(self.class_labels.index(class_label), dtype=torch.long)

        return x, y

def create_image_classification_dataloader(config: EasyDict, mode: str, validation_percentage: float = 0.2) -> Tuple[DataLoader, Dict[str, int]]:
    generator = ImageClassificationDataGenerator(config=config, mode=mode, validation_percentage=validation_percentage)
    dataloader = DataLoader(dataset=generator,
                            batch_size=config.learning.batch_size,
                            shuffle=config.learning.shuffle,
                            drop_last=config.learning.drop_last,
                            num_workers=2)

    return dataloader 

import time

if __name__ == '__main__':
    config = EasyDict(yaml.safe_load(open('config/config.yaml')))  # Load config file

    start_time = time.time()
    dataloader = create_image_classification_dataloader(config=config, mode='train') # Create dataloader
    dataloader2 = create_image_classification_dataloader(config=config, mode='val') # Create dataloader
    
    end_time = time.time()
    print(f"Time taken to create dataloader: {end_time - start_time} seconds")

    for i, (x, y) in enumerate(dataloader): # Iterate over the dataset
        start_time = time.time()
        print(x.shape, y.shape)
        end_time = time.time()
        print(f"Time taken to load batch {i}: {end_time - start_time} seconds")
        