import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

import informations as info


def create_folder(folderpath: str) -> str:
    if not os.path.exists(folderpath):
        print(f'create folder: {folderpath}')
        os.mkdir(folderpath)


def create_all_folder(dst_path: str) -> None:
    create_folder(dst_path)
    for label in info.LABELS:
        create_folder(os.path.join(dst_path, label))
    
    for label in info.LABELS:
        for background in info.BACKGROUND:
            create_folder(os.path.join(dst_path, label, background))


def main(mode: str) -> None:
    dst_path = os.path.join(info.DST_PATH, f'{mode}_{info.IMAGE_SIZE}')
    print(f'{dst_path=}')
    create_all_folder(dst_path)
    
    transform = transforms.Compose([
            transforms.Resize((info.IMAGE_SIZE, info.IMAGE_SIZE)),
            transforms.ToTensor(),
    ])

    data = pd.read_csv(os.path.join(info.DST_PATH, f'{mode}_item.csv'))
    for i in tqdm(range(len(data))):
        _, image_path, label, background = data.loc[i]
        img = Image.open(image_path)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        x = transform(img)

        dst_file = os.path.join(dst_path, label, background, f'{i}.jpg')
        transforms.ToPILImage()(x).save(dst_file)


if __name__ == '__main__':
    for mode in ['train', 'test', 'val']:
        print(f'{mode = }')
        main(mode=mode)


