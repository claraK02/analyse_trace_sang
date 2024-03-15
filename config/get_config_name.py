from easydict import EasyDict


MODEL_IMPLEMENTED = ['resnet']


def get_config_name(config: EasyDict) -> str:
    """
    Get an explicit name according to the configuration.

    Args:
        config (EasyDict): The configuration object.

    Raises:
        ValueError: If the model name is not in MODEL_IMPLEMENTED.

    Returns:
        str: The explicit name according to the configuration.
    """
    if config.model.name not in MODEL_IMPLEMENTED:
        raise ValueError(f'Expected model name in {MODEL_IMPLEMENTED} '
                         f'but found {config.model.name}')

    if config.model.name == 'resnet':
        return get_config_name_resnet(config)


def get_config_name_resnet(config: EasyDict) -> str:
    """
    Get the configuration name for the ResNet model.

    Args:
        config (EasyDict): The configuration object.

    Returns:
        str: The configuration name.
    """
    resnet: EasyDict = config.model.resnet
    model: str = 'resnet'
    freeze_param: bool = resnet.freeze_resnet

    if 'resume_training' in resnet.keys() and resnet.resume_training.do_resume:
        model = 'retrain_resnet'
        freeze_param = resnet.resume_training.freeze_param
    
    if not freeze_param:
        name = f'{model}_allw_img{config.data.image_size}'
    else:
        name = f'{model}_img{config.data.image_size}'
    
    return name
    
