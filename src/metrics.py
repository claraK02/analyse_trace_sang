import torch

def accuracy_one_hot(y_true, y_pred):
    """
    Calculates the accuracy of the model between one hot vectors and one hot vectors
    """

    return (y_true == y_pred).float().mean()

def accuracy_pytorch(y_true, y_pred):
    """
    Calculates the accuracy of the model between one hot vectors(y_pred) and indices(y_true)
    """
    _, predicted_class = torch.max(y_pred, 1)
    return (predicted_class == y_true).float().mean()


if __name__ == '__main__':
    batch_size = 3
    num_classes = 3
    y_true = torch.randint(num_classes, size=(batch_size,)) #shape: torch.Size([10])
    print('y_true',y_true)
    y_pred = torch.randint(num_classes, size=(batch_size,)) #shape: torch.Size([10])
    print('y_pred',y_pred)
    y_pred_onehot = torch.nn.functional.one_hot(y_pred, num_classes=num_classes) #shape: torch.Size([10, 3])
    print('accuracy pytorch',accuracy_pytorch(y_true, y_pred_onehot))
    print('accuracy one hot',accuracy_one_hot(y_true, y_pred))