import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname as up
import matplotlib.pyplot as plt

sys.path.append(up(up(os.path.abspath(__file__))))

from src.dataloader.labels import LABELS, BACKGROUND


def get_distribution(mode: str,
                     datapath: str=os.path.join('data', 'data_labo')
                     ) -> np.ndarray[float]:
    datapath = os.path.join(datapath, f'{mode}_128')
    labels_distribution = []
    for label in LABELS:
        num_data = 0
        for bg in BACKGROUND:
            num_data += len(os.listdir(os.path.join(datapath, label, bg)))
        labels_distribution.append(num_data)

    N = sum(labels_distribution)
    for i in range(len(labels_distribution)):
        labels_distribution[i] /= N
    
    return np.array(labels_distribution)


def plot_distrib(labels_distribution: list[int]):
    total_classes = len(labels_distribution)
    classes_index = range(1, total_classes + 1)

    plt.figure(figsize=(15, 12))
    plt.bar(classes_index, labels_distribution)

    plt.title('Distribution des classes')
    plt.xlabel('Classe')
    plt.ylabel('Proportion')

    plt.xticks(classes_index, LABELS, rotation=45, ha='right')

    plt.savefig('distribution.png')


def plot_3distribution(distribution1, distribution2, distribution3) -> None:
    classes_index = np.arange(1, 19)
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(classes_index - bar_width, distribution1, bar_width, label='Distribution 1')
    ax.bar(classes_index, distribution2, bar_width, label='Distribution 2')
    ax.bar(classes_index + bar_width, distribution3, bar_width, label='Distribution 3')

    ax.set_title('Comparaison de trois distributions')
    ax.set_xlabel('Classe')
    ax.set_ylabel('Proportion')
    ax.set_xticks(classes_index)
    ax.set_xticklabels(classes_index)
    ax.legend()

    plt.tight_layout()
    plt.savefig('distribution_train_val_test.png')


if __name__ == '__main__':
    train = get_distribution(mode='train')
    val = get_distribution(mode='val')
    test = get_distribution(mode='test')

    plot_distrib((train + val + test) / 3)
    # plot_3distribution(train, val, test)

    # print('label, train, val, test')
    # for i in range(len(info.LABELS)):
    #     print(f'{i + 1:2}: {train[i]:.2f} {val[i]:.2f} {test[i]:.2f}')