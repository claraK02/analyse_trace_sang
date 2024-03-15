import numpy as np

import torch
from torch import Tensor
from torchmetrics import Accuracy, F1Score, Precision, Recall


class Accuracy_per_class:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> list[float]:
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
            per_label_accuracies.append((correct / total).item() if total > 0 else 0)

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


class Metrics:
    def __init__(self,
                 num_classes: int,
                 run_argmax_on_y_true: bool=True,
                 run_acc_per_class: bool=False) -> None:
        """
        Initializes the Metrics class.

        Args:
            num_classes (int): The number of classes.
            run_argmax_on_y_true (bool, optional): Whether to run argmax on y_true. Defaults to True.
            run_acc_per_class (bool, optional): Whether to run accuracy per class. Defaults to False.
        
        Returns:
            None
        """
        micro = {'task': 'multiclass', 'average': 'micro', 'num_classes': num_classes}
        macro = {'task': 'multiclass', 'average': 'macro', 'num_classes': num_classes}

        self.metrics = {'acc micro': Accuracy(**micro),
                        'acc macro': Accuracy(**macro) ,
                        'precission macro': Precision(**macro),
                        'recall macro': Recall(**macro),
                        'f1-score macro': F1Score(**macro)}
        
        self.metrics_onehot = {'top k micro': Accuracy(top_k=3, **micro),
                               'top k macro': Accuracy(top_k=3, **macro)}
        if run_acc_per_class:
            self.metrics_per_class = Accuracy_per_class(num_classes=num_classes)
            self.num_metrics = len(self.metrics_onehot) + len(self.metrics) + num_classes
            self.metrics_name = list(self.metrics_onehot.keys()) + list(self.metrics.keys()) + self.metrics_per_class.get_metrics_name()
        else:
            self.num_metrics = len(self.metrics_onehot) + len(self.metrics)
            self.metrics_name = list(self.metrics_onehot.keys()) + list(self.metrics.keys())
        
        self.run_argmax_on_y_true = run_argmax_on_y_true
        self.run_acc_per_class = run_acc_per_class
    
    def compute(self,
                y_pred: Tensor,
                y_true: Tensor
                ) -> np.ndarray:
        """
        Computes all the metrics.

        Args:
            y_pred (Tensor): The predicted values with shape (B, num_classes).
            y_true (Tensor): The true values with shape (B, num_classes).

        Returns:
            np.ndarray: The computed metrics values.
        """
        metrics_value = []
        if self.run_argmax_on_y_true:
            y_true = torch.argmax(y_true, dim=-1)

        for metric in self.metrics_onehot.values():
            metrics_value.append(metric(y_pred, y_true).item())

        y_pred = torch.argmax(y_pred, dim=-1)
        
        for metric in self.metrics.values():
            metrics_value.append(metric(y_pred, y_true).item())
        
        if self.run_acc_per_class:
            metric_per_class = self.metrics_per_class(y_pred, y_true)
            metrics_value += metric_per_class

        return np.array(metrics_value)
    
    def get_names(self) -> list[str]:
        """
        Returns the names of the metrics.

        Returns:
            list[str]: The names of the metrics.
        """
        return self.metrics_name
    
    def init_metrics(self) -> np.ndarray:
        """
        Initializes the metrics.

        Returns:
            np.ndarray: The initialized metrics.
        """
        return np.zeros(self.num_metrics)
    
    def to(self, device: torch.device) -> None:
        """
        Moves the metrics to the specified device.

        Args:
            device (torch.device): The device to move the metrics to.
        """
        for key in self.metrics_onehot.keys():
            self.metrics_onehot[key] = self.metrics_onehot[key].to(device)
        for key in self.metrics.keys():
            self.metrics[key] = self.metrics[key].to(device)

    def get_info(self, metrics_value: np.ndarray) -> str:
        """
        Get information about the metrics values.

        Args:
            metrics_value (np.ndarray): An array of metrics values.

        Raises:
            ValueError: If the length of metrics_value is not equal to num_metrics.

        Returns:
            str: A string containing the metrics names and their corresponding values.
        """
        if len(metrics_value) != self.num_metrics:
            raise ValueError(f'metrics_value doesnt have the same length as num_metrics.',
                             f'{len(metrics_value) = } and {self.num_metrics = }')
        
        output = 'Metrics \t: Values\n'
        output += '-' * 15 + ' | ' + '-' * 4 + '\n'
        for i, metric_name in enumerate(self.metrics_name):
            output += f'{metric_name[:14]}\t: {metrics_value[i]:.2f}\n'
        return output


if __name__ == '__main__':
    batch_size = 32
    num_classes = 19
    y_pred = torch.rand(size=(batch_size, num_classes))
    y_true = torch.randint(num_classes, size=(batch_size,))
    # print('y_true', y_true.shape)
    # print(y_true)
    # print('y_pred', y_pred.shape)
    # print(y_pred)
    
    print(Accuracy_per_class(num_classes=num_classes)(y_pred, y_true))
    
    metrics = Metrics(num_classes=num_classes, run_argmax_on_y_true=False, run_acc_per_class=True)
    metrics_value = metrics.compute(y_pred, y_true)
    print(metrics.get_info(metrics_value))