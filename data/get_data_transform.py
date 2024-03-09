# Import necessary libraries
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

# Import custom module
import informations as info

# Function to create a folder if it doesn't exist
def create_folder(folderpath: str) -> str:
    if not os.path.exists(folderpath):
        print(f'create folder: {folderpath}')
        os.mkdir(folderpath)

# Function to create all necessary folders for each label and background
def create_all_folder(dst_path: str) -> None:
    create_folder(dst_path)
    for label in info.LABELS:
        create_folder(os.path.join(dst_path, label))
    
    for label in info.LABELS:
        for background in info.BACKGROUND:
            create_folder(os.path.join(dst_path, label, background))

# Main function to process the images
def main(mode: str, bakground_wanted: bool = False) -> None:
    """
    Main function to transform and save images.

    Args:
        mode (str): The mode of the data (e.g., train, test, validation).

    Returns:
        None
    """
    # Define the destination path
    print("DESTINATION PATH:", info.DST_PATH)
    dst_path = os.path.join(info.DST_PATH, f'{mode}_{info.IMAGE_SIZE}')
    print(f'{dst_path=}')
    # Create all necessary folders
    create_all_folder(dst_path)
    
    # Define the image transformation: resize and convert to tensor
    transform = transforms.Compose([
            transforms.Resize((info.IMAGE_SIZE, info.IMAGE_SIZE)),
            transforms.ToTensor(),
    ])

    # Read the data from csv file
    data = pd.read_csv(os.path.join('data', f'{mode}_item.csv'))
    # Loop over all images
    for i in tqdm(range(len(data))):
        _, image_path, label, background = data.loc[i]
        print("IMAGE PATH:", image_path)
        # Open the image
        img = Image.open(image_path)
        # If the image is RGBA, convert it to RGB
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        # Apply the transformation
        x = transform(img)

        # Define the destination file path

        if bakground_wanted: #si on traite les données de labo
            dst_file = os.path.join(dst_path, label, background, f'{i}.jpg')
        else: #si on traite les données de terrain on connait pas le background !!!
            dst_file = os.path.join(dst_path, label, f'{i}.jpg')

        # Save the transformed image
        transforms.ToPILImage()(x).save(dst_file)
        print("the image was saved in", dst_file)

# If the script is run directly, process images for 'train', 'test', and 'val' modes
if __name__ == '__main__':
    for mode in ['train', 'test', 'val']:
        print(f'{mode = }')
        main(mode=mode)