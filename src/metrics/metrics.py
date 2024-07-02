import os
import sys
import numpy as np
from os.path import dirname as up

import torch
from torch import Tensor
from torchmetrics import Accuracy, F1Score, Precision, Recall

sys.path.append(up(up(up(os.path.abspath(__file__)))))

from src.metrics import accuracy_per_classes, silancy_metrics
                
class Metrics:
    def __init__(self,
                 num_classes: int,
                 run_argmax_on_y_true: bool = True,
                 run_acc_per_class: bool = False,
                 run_silancy_metrics: bool = False,
                 ) -> None:
        """
        Initializes the Metrics class.

        Args:
            num_classes (int): The number of classes.
            run_argmax_on_y_true (bool, optional): Whether to run argmax on y_true. Defaults to True.
            run_acc_per_class (bool, optional): Whether to run accuracy per class. Defaults to False.
            run_silancy_metrics (bool, optional): Whether to run silancy metrics. Defaults to False.
        """

        micro = {'task': 'multiclass', 'average': 'micro', 'num_classes': num_classes}
        macro = {'task': 'multiclass', 'average': 'macro', 'num_classes': num_classes}

        self.metrics = {'acc micro': Accuracy(**micro),
                        'acc macro': Accuracy(**macro) ,
                        'precission macro': Precision(**macro),
                        'recall macro': Recall(**macro),
                        'f1-score macro': F1Score(**macro)}
        
        self.metrics_onehot = {'top k micro': Accuracy(top_k=2, **micro),
                               'top k macro': Accuracy(top_k=2, **macro)}
        
        self.num_metrics: int = len(self.metrics_onehot) + len(self.metrics)
        self.metrics_name: list[str] = list(self.metrics_onehot.keys())

        if run_silancy_metrics:
            self.metrics_silancy = silancy_metrics.Silancy_Metrics()
            self.num_metrics += 3
            self.metrics_name += self.metrics_silancy.get_metrics_name()
        
        self.metrics_name += list(self.metrics.keys()) 

        if run_acc_per_class:
            self.metrics_per_class = accuracy_per_classes.Accuracy_per_class(num_classes=num_classes)
            self.num_metrics += num_classes
            self.metrics_name += self.metrics_per_class.get_metrics_name()
        
        self.run_argmax_on_y_true = run_argmax_on_y_true
        self.run_acc_per_class = run_acc_per_class
        self.run_silancy_metrics = run_silancy_metrics
    
    def compute(self,
                y_pred: Tensor,
                y_true: Tensor,
                o_pred: Tensor = None,
                ) -> np.ndarray:
        """
        Computes all the metrics.

        Args:
            y_pred (Tensor): The predicted values with shape (B, num_classes).
            y_true (Tensor): The true values with shape (B, num_classes).
            o_pred (Tensor, optional): The predicted probability given as input the image x with the mask. Defaults to None.

        Returns:
            np.ndarray: The computed metrics values.
        """
        metrics_value = []
        if self.run_argmax_on_y_true:
            y_true = torch.argmax(y_true, dim=-1)

        for metric in self.metrics_onehot.values():
            metrics_value.append(metric(y_pred, y_true).item())

        if self.run_silancy_metrics:
            metrics_value += self.metrics_silancy.compute(y_pred, o_pred)

        y_pred = torch.argmax(y_pred, dim=-1)
        
        for metric in self.metrics.values():
            metrics_value.append(metric(y_pred, y_true).item())
        
        if self.run_acc_per_class:
            metrics_value += self.metrics_per_class.compute(y_pred, y_true)

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
    y_pred = torch.softmax(y_pred, dim=1)
    y_true = torch.randint(num_classes, size=(batch_size,))
    epsilon = torch.rand(size=(batch_size, num_classes)) * 0.1
    o_pred = y_pred + epsilon
    o_pred = torch.softmax(o_pred, dim=1)
    
    metrics = Metrics(num_classes=num_classes,
                      run_argmax_on_y_true=False,
                      run_acc_per_class=True,
                      run_silancy_metrics=True)
    metrics_value = metrics.compute(y_pred, y_true, o_pred=o_pred)
    print(metrics.get_info(metrics_value))