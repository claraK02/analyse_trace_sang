import torch


def get_device(device_config: str) -> torch.device:
    """ get device: cuda or cpu """
    if torch.cuda.is_available() and device_config == 'cuda':
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device