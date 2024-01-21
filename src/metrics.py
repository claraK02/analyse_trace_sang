import torch

def accuracy_one_hot(y_true, y_pred):
    """
    Calculates the accuracy of the model between one hot vectors and one hot vectors
    """

    return (y_true == y_pred).mean()

def accuracy_pytorch(y_true, y_pred):
    """
    Calculates the accuracy of the model between one hot vectors(y_pred) and indices(y_true)
    """
    _, predicted_class = torch.max(y_pred, 1)
    return (predicted_class == y_true).float().mean()