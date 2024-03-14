import os
import sys
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from os.path import dirname as up

sys.path.append(up(up(up(os.path.abspath(__file__)))))

from src.dataloader.labels import LABELS


def create_folder(folderpath: str) -> str:
    if not os.path.exists(folderpath):
        print(f'create folder: {folderpath}')
        os.mkdir(folderpath)


def create_all_folder(dst_path: str) -> None:
    create_folder(dst_path)
    for label in LABELS:
        create_folder(os.path.join(dst_path, label))


def main(mode: str,
         dst_path: str,
         image_size: int,
         item_path: str) -> None:
    
    create_folder(folderpath=dst_path)
    dst_path = os.path.join(dst_path, f'{mode}_{image_size}')
    print(f'{dst_path=}')
    create_all_folder(dst_path)
    
    # Define the image transformation: resize and convert to tensor
    transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
    ])

    # Read the data from csv file
    data = pd.read_csv(os.path.join(item_path, f'{mode}_item.csv'))

    # Loop over all images
    for i in tqdm(range(len(data))):
        line = data.iloc[i]
        img = Image.open(line['imagepath'])
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        x = transform(img)

        dst_file = os.path.join(dst_path, line['label'], f"{line['item']}.jpg")

        transforms.ToPILImage()(x).save(dst_file)


if __name__ == '__main__':
    data_real_path: str = os.path.join('data', 'data_real')
    item_path: str = os.path.join('data', 'process_real_data')

    for mode in ['train', 'test', 'val', 'all']:
        print(f'{mode = }')
        main(mode=mode,
             dst_path=data_real_path,
             image_size=256,
             item_path=item_path)
