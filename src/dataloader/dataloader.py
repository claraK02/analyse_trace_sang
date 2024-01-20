import os
from easydict import EasyDict
from typing import List, Tuple, Dict
from PIL import Image

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
import yaml
from easydict import EasyDict
from pprint import pprint

class ImageClassificationDataGenerator(Dataset):
    def __init__(self, config: EasyDict, mode: str) -> None:
        assert mode in ['train', 'val', 'test'], f"Error, expected mode is train, val or test but found '{mode}'"
        self.mode = mode
        self.transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            # Add other transformations as needed
        ])

        # We create one list of paths per class

        # Get the list of classes! they are the folder names in data_path
        self.class_labels = sorted(os.listdir('data'))
        print(f"Found {len(self.class_labels)} classes: {self.class_labels}")
        
        #get all the paths of all the images in all the folders or subfolders or subsubfolders or subsubsubfolders in each class

        # Get all the paths of all the images in all the folders or subfolders or subsubfolders or subsubsubfolders in each class
        self.class_paths = {class_label: [] for class_label in self.class_labels}
        for class_label in self.class_labels:
            print(f"Searching for images in {class_label}..." )
            for root, dirs, files in os.walk(os.path.join('data', class_label)):
                print(f"Found {len(files)} images in {root}")
                for file in files:
                    if file.endswith(('.png', '.jpg', '.jpeg','.JPG')):  # add or remove file extensions as needed
                        path = os.path.join(root, file)
                        if "Retouches" in path: #on ne prend pas les images non retouchées
                            self.class_paths[class_label].append(path)

        #print number of images in each class
        for class_label in self.class_labels:
            print(f"Found {len(self.class_paths[class_label])} images for class {class_label}")
        
        # Flatten all image paths and their corresponding class labels into two separate lists
        self.all_image_paths = []
        self.all_image_labels = []
        for class_label, paths in self.class_paths.items():
            self.all_image_paths.extend(paths)
            self.all_image_labels.extend([class_label] * len(paths))
        print(f"The {self.mode} generator was created")

    def __len__(self) -> int:
        return len(self.all_image_paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
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

def create_image_classification_dataloader(config: EasyDict, mode: str) -> Tuple[DataLoader, Dict[str, int]]:
    generator = ImageClassificationDataGenerator(config=config, mode=mode)
    dataloader = DataLoader(dataset=generator,
                            batch_size=config.learning.batch_size,
                            shuffle=config.learning.shuffle,
                            drop_last=config.learning.drop_last)
    return dataloader, None  # Image classification typically does not use vocabulary

if __name__ == '__main__':
   

    config = EasyDict(yaml.safe_load(open('config/config.yaml')))  # Load config file
    dataloader, _ = create_image_classification_dataloader(config=config, mode='train') # Create dataloader

    for x, y in dataloader: # Iterate over the dataset
        print(x.shape, y.shape)
        print(y)
        break

    #plot images of the batch, first convert them from tensor to numpy

    x = x.numpy()
    y = y.numpy()
    fig = plt.figure(figsize=(10, 10))
    #batch size is 2
    for i in range(2):
        ax = fig.add_subplot(1, 2, i + 1)
        ax.imshow(np.transpose(x[i], (1, 2, 0)))
        ax.set_title(y[i])
        ax.axis('off')
    plt.show()


    print("Done!")