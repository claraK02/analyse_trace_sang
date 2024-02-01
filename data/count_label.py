import os
from icecream import ic
import informations as info

DATA_PATH = ['data/train_128', 'data/val_128', 'data/test_128']

labels_bg: dict[str, int] = {}

src = 'data/data_retouche/train_128'
for label in info.LABELS:
    labels_bg[label] = 0
    for background in info.BACKGROUND:
        path = os.path.join(src, label, background)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        if len(os.listdir(path)) != 0:
            labels_bg[label] += 1

ic(labels_bg)
labels = list(filter(lambda key: labels_bg[key] != 0, labels_bg))
print(labels)