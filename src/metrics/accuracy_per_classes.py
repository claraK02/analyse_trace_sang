import numpy as np
from torch import Tensor


class Accuracy_per_class:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes

    def compute(self, y_pred: Tensor, y_true: Tensor) -> list[float]:
        """
        Compute the accuracy per class.

        Args:
            y_pred (Tensor): The predicted labels.
            y_true (Tensor): The true labels.

        Returns:
            list[float]: The accuracy per class.
        """
        # Convert y_true to an integer tensor
        y_true = y_true.long()

        per_label_accuracies = []
        
        for label in range(self.num_classes):
            # Compute the accuracy for each class
            correct = (y_pred[y_true == label] == label).sum()
            total = (y_true == label).sum()
            per_label_accuracies.append((correct / total).item() if total > 0 else np.nan)

        return per_label_accuracies
    
    def get_metrics_name(self) -> list[str]:
        """
        Get the names of the metrics.

        Returns:
            list[str]: The names of the metrics.
        """
        metrics_name = []
        for i in range(self.num_classes):
            metrics_name.append(f'acc class n°{i + 1}')
        return metrics_name


if __name__ == '__main__':
    import torch

    batch_size = 32
    num_classes = 18
    y_pred = torch.rand(size=(batch_size, num_classes))
    y_true = torch.randint(num_classes, size=(batch_size,))

    metrics = Accuracy_per_class(num_classes=num_classes)
    
    print(metrics(y_pred, y_true))
    print(metrics.get_metrics_name())