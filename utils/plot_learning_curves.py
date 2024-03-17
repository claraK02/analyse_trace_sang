import os
import numpy as np
import matplotlib.pyplot as plt


def print_loss_and_metrics(train_loss: float,
                           val_loss: float,
                           metrics_name: list[str],
                           train_metrics: list[float],
                           val_metrics: list[float]) -> None:
    """
    Prints the loss and metrics for train and validation.

    Args:
        train_loss (float): The loss value for the training set.
        val_loss (float): The loss value for the validation set.
        metrics_name (list[str]): A list of metric names.
        train_metrics (list[float]): A list of metric values for the training set.
        val_metrics (list[float]): A list of metric values for the validation set.
    """
    print(f"{train_loss = }")
    print(f"{val_loss = }")
    for i in range(len(metrics_name)):
        print(f"{metrics_name[i]} -> train: {train_metrics[i]:.3f}   val:{val_metrics[i]:.3f}")


def save_learning_curves(path: str) -> None:
    result, names = get_result(path)

    epochs = result[:, 0]
    for i in range(1, len(names), 2):
        train_metrics = result[:, i]
        val_metrics = result[:, i + 1]
        plt.plot(epochs, train_metrics)
        plt.plot(epochs, val_metrics)
        plt.title(names[i])
        plt.xlabel('epoch')
        plt.ylabel(names[i])
        plt.legend(names[i:])
        plt.grid()
        plt.savefig(os.path.join(path, names[i] + '.png'))
        plt.close()


def get_result(path: str) -> tuple[list[float], list[str]]:
    train_log_name = 'train_log.csv'
    if train_log_name not in os.listdir(path):
        train_log_name = 'train_real_log.csv'
    
    with open(os.path.join(path, train_log_name), 'r') as f:
        names = f.readline()[:-1].split(',')
        result = []
        for line in f:
            result.append(line[:-1].split(','))

        result = np.array(result, dtype=float)
    f.close()
    return result, names


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str,
                        help="log to plot learning curves")
    args = parser.parse_args()

    save_learning_curves(path=args.path)
