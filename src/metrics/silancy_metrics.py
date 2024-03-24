from typing import Callable
import torch
from torch import Tensor



class Silancy_Metrics:
    def __init__(self) -> None:
        """
        Initialize the SilancyMetrics class witch compute the Silancy metrics.
        See https://arxiv.org/pdf/2301.07002.pdf for more details.
        """
        self.smooth = 1e-6
        self.metrics: dict[str, Callable] = {
            'avg_drop': self.__get_average_drop,
            'avg_increase': self.__get_average_increase,
            'avg_gain': self.__get_average_gain
        }
        
    def compute(self, y_pred: Tensor, o_pred: Tensor) -> list[float]:
        """
        Calculate the metrics for the predicted and observed tensors.

        Args:
            y_pred (Tensor): The predicted probability given as input the image x 
                tensor with shape (batch_size, num_classes).
            o_pred (Tensor): The predicted probability given as input the image x with the mask
                tensor with shape (batch_size, num_classes).

        Raises:
            ValueError: If the shape of y_pred is not 2D.
            ValueError: If the shape of o_pred is not 2D.

        Returns:
            list[float]: A list of metric values calculated for the predicted and observed tensors.
        """
        if len(y_pred.shape) != 2:
            raise ValueError(f'y_pred must be a 2D tensor, but got {len(y_pred.shape)}D tensor.')
        if len(o_pred.shape) != 2:
            raise ValueError(f'o_pred must be a 2D tensor, but got {len(o_pred.shape)}D tensor.')
        
        y_argmax = torch.argmax(y_pred, dim=1)
        line_number = torch.arange(len(y_argmax))
        p = y_pred[line_number, y_argmax]
        o = o_pred[line_number, y_argmax]

        return list(map(lambda metric: metric(p, o), self.metrics.values()))

    def get_metrics_name(self) -> list[str]:
        """ Get the names of the metrics. """
        return list(self.metrics.keys())

    def __get_average_drop(self, p: Tensor, o: Tensor) -> float:
        """
        Calculate the average drop.

        Args:
            p (Tensor): predicted probability given as input the image x
            o (Tensor): predicted probability given as input the image x with the mask

        Returns:
            float: The average drop.
        """
        clamp = (p - o).clamp(min=0)
        divide = clamp / (p + self.smooth)
        return divide.mean().item()
    
    def __get_average_increase(self, p: Tensor, o: Tensor) -> float:
        """
        Calculate the average increase.

        Args:
            p (Tensor): predicted probability given as input the image x
            o (Tensor): predicted probability given as input the image x with the mask

        Returns:
            float: The average drop.
        """
        clamp = (o - p).clamp(min=0)
        return clamp.mean().item()
    
    def __get_average_gain(self, p: Tensor, o: Tensor) -> float:
        """
        Calculate the average gain.

        Args:
            p (Tensor): predicted probability given as input the image x
            o (Tensor): predicted probability given as input the image x with the mask

        Returns:
            float: The average drop.
        """
        clamp = (o - p).clamp(min=0)
        divide = clamp / (1 - p + self.smooth)
        return divide.mean().item()


if __name__ == '__main__':
    batch_size = 32
    num_classes = 18
    y_pred = torch.rand(size=(batch_size, num_classes))
    epsilon = torch.rand(size=(batch_size, num_classes)) * 0.5
    o_pred = y_pred + epsilon
    y_pred = torch.softmax(y_pred, dim=1)
    o_pred = torch.softmax(o_pred, dim=1)

    metrics = Silancy_Metrics()
    
    print(metrics(y_pred=y_pred, o_pred=o_pred))
    print(metrics.get_metrics_name())