import torch

LABELS = [
    "1- Modèle Traces passives",
    "2- Modèle Goutte à Goutte",
    "3- Modèle Transfert par contact",
    "4- Modèle Transfert glissé",
    "5- Modèle Altération par contact",
    "6- Modèle Altération glissée",
    "7- Modèle d'Accumulation",
    "8- Modèle Coulée",
    "9- Modèle Chute de volume",
    "10- Modèle Sang Propulsé",
    "11- Modèle d'éjection",
    "12- Modèle Volume impacté",
    "13- Modèle Imprégnation",
    "14- Modèle Zone d'interruption",
    "15- Modèle Modèle d'impact",
    "16- Modèle Foyer de modèle d'impact",
    "17- Modèle Trace gravitationnelle",
    "18- Modèle Sang expiré",
]
BACKGROUND = ["carrelage", "papier", "bois", "lino"]

DOMAIN = ['data_labo', "new_real_data"]


def get_label_prediction(y_pred: torch.Tensor) -> list[tuple[int, str]]:
    """
    Get the label predictions from the model's output tensor.

    Args:
        y_pred (torch.Tensor): The model's output tensor containing the predicted labels.

    Raises:
        ValueError: If the shape of y_pred does not match the expected shape.

    Returns:
        list[tuple[int, str]]: A list of tuples containing the predicted label index and its corresponding label string.
    """
    if y_pred.shape != (len(y_pred), len(LABELS)):
        raise ValueError(f'Expected y_pred with shape of {len(LABELS)}'
                         f'but found {y_pred.shape}')
    
    pred = torch.argmax(y_pred, dim=-1)
    output: list[tuple[int, str]] = []
    for i in range(len(pred)):
        output.append((pred[i].item(), LABELS[pred[i].item()]))
    return output


def get_topk_prediction(y_pred: torch.Tensor,
                        k: int = 3
                        ) -> list[list[tuple[int, str, float]]]:
    """
    Get the top-k predictions from the given tensor.

    Args:
        y_pred (torch.Tensor): The input tensor containing the predictions.
        k (int, optional): The number of top predictions to retrieve. Defaults to 3.

    Returns:
        list[list[tuple[int, str, float]]]: A list of lists, where each inner list contains tuples
        representing the top-k predictions for each input in y_pred. Each tuple contains the index,
        label, and value of the prediction.
    """
    topk_values, topk_indices = torch.topk(y_pred, k=3)

    output: list[list[tuple[int, str, float]]] = []
    for i in range(len(y_pred)):
        pred_i: list[tuple[int, str, float]] = []
        for j in range(k):
            pred_i.append((topk_indices[i, j].item(), LABELS[topk_indices[i, j].item()], topk_values[i, j].item()))
        output.append(pred_i)
    
    return output


if __name__ == '__main__':
    y_pred = torch.randn(4, len(LABELS))
    y_pred = torch.nn.functional.softmax(y_pred, dim=-1)

    print(get_label_prediction(y_pred))
    print(get_topk_prediction(y_pred))
