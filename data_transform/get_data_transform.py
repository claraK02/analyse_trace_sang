import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

from torchvision import transforms


DATA_PATH = 'data'
DATA_TRANSFORM_PATH = 'data_transform'
IMAGE_SIZE = 128

LABEL = ['1- Modèle Traces passives', '4- Modèle Transfert glissé']
BACKGROUND = ['carrelage', 'papier', 'bois', 'lino']
NEW_LABEL = ['traces_passive', 'transfert_glisse']


def create_folder(folderpath: str) -> str:
    if not os.path.exists(folderpath):
        print(f'create folder: {folderpath}')
        os.mkdir(folderpath)


def create_all_folder(dst_path: str) -> None:
    create_folder(dst_path)
    for label in NEW_LABEL:
        create_folder(os.path.join(dst_path, label))
    
    for label in NEW_LABEL:
        for background in BACKGROUND:
            create_folder(os.path.join(dst_path, label, background))


def get_new_label(old_label: str) -> str:
    return NEW_LABEL[LABEL.index(old_label)]


def main(mode: str) -> None:
    dst_path = os.path.join(DATA_TRANSFORM_PATH, f'{mode}_{IMAGE_SIZE}')
    create_all_folder(dst_path)
    
    transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
    ])

    data = pd.read_csv(os.path.join(DATA_PATH, f'{mode}_item.csv'))
    for i in tqdm(range(len(data))):
        _, image_path, label, background = data.loc[i]
        img = Image.open(image_path)
        x = transform(img)

        dst_file = os.path.join(dst_path, get_new_label(label), background, f'{i}.jpg')
        transforms.ToPILImage()(x).save(dst_file)


if __name__ == '__main__':
    for mode in ['train', 'test', 'val']:
        print(f'{mode = }')
        main(mode=mode)


