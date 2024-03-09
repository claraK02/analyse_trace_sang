# This script is used to split the dataset into training, validation, and testing sets.
# It first gathers all the image data, then splits it according to the specified proportions,
# and finally saves each set into a separate CSV file.

import os
import random
from tqdm import tqdm
import informations as info

# Set the seed for the random number generator to ensure reproducibility
random.seed(0)

# Define a type for the data: a list of tuples, where each tuple contains a string (the image path),
# a string (the label), and a string (the background)
DATA_TYPE = list[tuple[str, str, str]]
 
def get_data(datapath: str,
             class_labels: list[str],
             backgrounds: list[str],
             only_retouche: bool=True,
             background_wanted: bool=False,
             ) -> list[tuple[str, str, str]]:
    """
    Gather all the image data from the specified path.
    Each image is represented as a tuple containing the image path, the label, and the background.
    If `only_retouche` is True, only images with "Retouches" in their path are included.
    """
    all_image_paths = []
    all_image_labels = []
    all_image_backgrounds = []

    #print how many labels we have
    print(f'we have {len(class_labels)} labels')

    # Loop over all labels
    for class_label in class_labels:
        # Loop over all files in the directory corresponding to the current label
        for root, _, files in tqdm(os.walk(os.path.join(datapath, class_label)), desc='walk in data'):
            # Loop over all files
            for file in files:
                # Check if the file is an image
                if file.endswith(('.png', '.jpg', '.jpeg', '.JPG')):
                    path = os.path.join(root, file)
                    # Check if the image should be included
                    if (not only_retouche) or ("Retouches" in path):
                        print('Appending path to all_image_paths list: ', path)
                        all_image_paths.append(path)
                        all_image_labels.append(class_label)
                        # Loop over all backgrounds and check if the current image has this background
                        if background_wanted:
                            for bg in backgrounds:
                                if bg in path.lower():
                                    all_image_backgrounds.append(bg)

    # Combine the lists into a list of tuples and shuffle it
    if background_wanted:
        c = list(zip(all_image_paths, all_image_labels, all_image_backgrounds))
        random.shuffle(c)
    else:
        #we put 'papier' as background for all images
        c = list(zip(all_image_paths, all_image_labels, ['papier']*len(all_image_paths)))
        random.shuffle(c)
    
    print(f'We have {len(c)} images in total')
    return c


def split_data(data: DATA_TYPE,
               train_rate: float=0.8,
               val_rate: float=0.1,
               test_rate: float=0.1
               ) -> tuple[DATA_TYPE, DATA_TYPE, DATA_TYPE]:
    """
    Split the data into training, validation, and testing sets according to the specified proportions.
    """
    # Check if the proportions sum to 1
    if train_rate + val_rate + test_rate != 1:
        raise ValueError(f'train + val + test rate must be equal to 1,',
                         f'but is equal to {train_rate + val_rate + test_rate}.')

    n = len(data)
    split_1 = int(n * train_rate)
    split_2 = int(n * (train_rate + val_rate))

    train_data = data[:split_1]
    val_data = data[split_1 : split_2]
    test_data = data[split_2:]

    return train_data, val_data, test_data


def save_data(data: DATA_TYPE, mode: str, datapath: str) -> None:
    """
    Save the data into a CSV file.
    The file is saved in the specified path and its name is {mode}_item.csv.
    """
    output = ''
    for i, (path, label, background) in enumerate(data):
            output += f'{i},{path},{label},{background}\n'
    
    dstpath = os.path.join(datapath, f'{mode}_item.csv')
    with open(file=dstpath, mode='w', encoding='utf8') as f:
        f.write(output)
        f.close()
    print(f'data save in {dstpath}')
        

if __name__ == '__main__':
    # Gather the data
    data = get_data(datapath=info.DATAPATH,
                    class_labels=info.LABELS_PATH, #ATTENTION c'était info.LABELS avant !!
                    backgrounds=info.BACKGROUND,
                    background_wanted=False, #si on traite les données de labo on met True sinon False !
                    only_retouche='retouche' in info.DATAPATH)
    print(data)
    print(len(data))

    # Define the modes
    MODE = ['train', 'val', 'test']

    # Split the data
    data_split = split_data(data)

    # Save the data for each mode
    for i in range(3):
        save_data(data=data_split[i],
                  mode=MODE[i],
                  datapath='data')