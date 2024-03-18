import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname as up
import matplotlib.pyplot as plt

sys.path.append(up(up(os.path.abspath(__file__))))

from src.dataloader.labels import LABELS, BACKGROUND


def get_distribution(mode: str,
                     datapath: str=os.path.join('data', 'data_labo'),
                     use_background: bool=True
                     ) -> np.ndarray[float]:
    """
    Get the distribution of data for each label.

    Args:
        mode (str): The mode of the data (e.g., 'train', 'val', 'test').
        datapath (str, optional): The path to the data directory. Defaults to 'data/data_labo'.
        use_background (bool, optional): Whether to include background data. Defaults to True.

    Returns:
        np.ndarray[float]: The distribution of data for each label.
    """
    datapath = os.path.join(datapath, f'{mode}_256')
    labels_distribution = []
    for label in LABELS:
        num_data = 0

        if use_background:
            for bg in BACKGROUND:
                num_data += len(os.listdir(os.path.join(datapath, label, bg)))
        else:
            num_data = len(os.listdir(os.path.join(datapath, label)))

        labels_distribution.append(num_data)
    
    return np.array(labels_distribution)


def plot_distrib(labels_distribution: list[int],
                 title: str='distribution.png'
                 ) -> None:
    """
    Plot the distribution of data for each label.

    Args:
        labels_distribution (list[int]): The distribution of data for each label.
        title (str, optional): The title of the plot. Defaults to 'distribution.png'.

    Returns:
        None
    """
    total_classes = len(labels_distribution)
    classes_index = range(1, total_classes + 1)

    plt.figure(figsize=(15, 12))
    plt.bar(classes_index, labels_distribution)

    plt.title('Distribution des classes')
    plt.xlabel('Classe')
    plt.ylabel('Proportion')

    # plt.xticks(classes_index, LABELS, rotation=45, ha='right')

    plt.savefig(os.path.join('asset', title))


def plot_3distribution(distribution1: np.ndarray[float],
                       distribution2: np.ndarray[float],
                       distribution3: np.ndarray[float],
                       title='distribution_train_val_test.png'
                       ) -> None:
    """
    Plot the comparison of three distributions.

    Args:
        distribution1 (np.ndarray[float]): The first distribution.
        distribution2 (np.ndarray[float]): The second distribution.
        distribution3 (np.ndarray[float]): The third distribution.
        title (str, optional): The title of the plot. Defaults to 'distribution_train_val_test.png'.

    Returns:
        None
    """
    classes_index = np.arange(1, 19)
    bar_width = 0.25

    _, ax = plt.subplots(figsize=(12, 6))

    ax.bar(classes_index - bar_width, distribution1, bar_width, label='train distribution')
    ax.bar(classes_index, distribution2, bar_width, label='val distribution')
    ax.bar(classes_index + bar_width, distribution3, bar_width, label='test distribution')

    ax.set_title('Comparaison de trois distributions')
    ax.set_xlabel('Classe')
    ax.set_ylabel('Proportion')
    ax.set_xticks(classes_index)
    ax.set_xticklabels(classes_index)
    ax.legend()

    plt.tight_layout()
    # plt.xticks(classes_index, LABELS, rotation=45, ha='right')
    plt.savefig(os.path.join('asset', title))


if __name__ == '__main__':
    train = get_distribution(mode='train')
    val = get_distribution(mode='val')
    test = get_distribution(mode='test')

    plot_distrib((train + val + test) / 3)
    plot_3distribution(train, val, test)

    train_real = get_distribution(mode='train',
                                  datapath=os.path.join('data', 'data_real'),
                                  use_background=False)
    val_real = get_distribution(mode='val',
                                datapath=os.path.join('data', 'data_real'),
                                use_background=False)
    test_real = get_distribution(mode='test',
                                 datapath=os.path.join('data', 'data_real'),
                                 use_background=False)
    plot_distrib((train_real + val_real + test_real) / 3,
                 title='distribution_real.png')
    plot_3distribution(train_real, val_real, test_real,
                       title='distribution_train_val_test_real.png')