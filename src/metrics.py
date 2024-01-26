import numpy as np
from typing import List

import torch
from torch import Tensor
from torchmetrics import Accuracy, F1Score, Precision, Recall


class Metrics:
    def __init__(self,
                 num_classes: int,
                 average: str='micro',
                run_argmax_on_y_true: bool=True) -> None:
        self.num_classes = num_classes
        self.metrics = {'acc': Accuracy(task='multiclass',
                                        average=average,
                                        num_classes=num_classes),
                        'precission': Precision(task='multiclass',
                                                average=average,
                                                num_classes=num_classes),
                        'recall': Recall(task='multiclass',
                                         average=average,
                                         num_classes=num_classes),
                        'f1-score': F1Score(task='multiclass',
                                            average=average,
                                            num_classes=num_classes)}
        
        self.num_metrics = len(self.metrics)
        self.metrics_name = list(self.metrics.keys())
        self.run_argmax_on_y_true = run_argmax_on_y_true
    
    def compute(self,
                y_pred: Tensor,
                y_true: Tensor
                ) -> np.ndarray:
        """ compute all the metrics 
        y_pred and y_true must have shape like (B, 2)
        """
        metrics_value = []
        y_pred = torch.argmax(y_pred, dim=-1)
        if self.run_argmax_on_y_true:
            y_true = torch.argmax(y_true, dim=-1)
        
        for metric in self.metrics.values():
            metrics_value.append(metric(y_pred, y_true).item())
        return np.array(metrics_value)
    
    def get_names(self) -> List[str]:
        return list(self.metrics.keys())
    
    def init_metrics(self) -> np.ndarray:
        return np.zeros(self.num_metrics)
    
    def to(self, device: torch.device) -> None:
        for key in self.metrics.keys():
            self.metrics[key] = self.metrics[key].to(device)


if __name__ == '__main__':
    batch_size = 3
    num_classes = 3
    y_pred = torch.rand(size=(batch_size, num_classes))
    y_true = torch.randint(num_classes, size=(batch_size,))
    print('y_true',y_true)
    print('y_pred',y_pred)
    
    metrics = Metrics(num_classes=num_classes, run_argmax_on_y_true=False)
    print(metrics.compute(y_pred, y_true))