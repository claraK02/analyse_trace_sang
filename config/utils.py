import os
import sys
import yaml
from datetime import datetime
from easydict import EasyDict
from os.path import dirname as up

sys.path.append(up(os.path.abspath(__file__)))

from get_config_name import get_config_name


def number_folder(path: str, name: str) -> str:
    """
    Finds a declination of a folder name so that the name is not already taken.

    Args:
        path (str): The path to the directory where the folders are located.
        name (str): The base name of the folder.

    Returns:
        str: The new folder name that is not already taken.
    """
    elements = os.listdir(path)
    last_index = -1
    for i in range(len(elements)):
        folder_name = name + str(i)
        if folder_name in elements:
            last_index = i
    return name + str(last_index + 1)


def train_logger(config: EasyDict,
                 metrics_name: list[str] = None,
                 logspath: str = 'logs',
                 train_log_name: str = 'train_log.csv'
                 ) -> str:
    """
    Creates a logs folder where we can find the config in confing.yaml and
    create train_log.csv which will contain the loss and metrics values.

    Args:
        config (EasyDict): The configuration object.
        metrics_name (list[str], optional): List of metric names. If None, it will be inferred from the config. Defaults to None.
        logspath (str, optional): Path to the logs folder. Defaults to 'logs'.
        train_log_name (str, optional): Name of the train log file. Defaults to 'train_log.csv'.

    Returns:
        str: The path to the created logging folder.
    """
    if not os.path.exists(logspath):
        os.makedirs(logspath)
    folder_name = number_folder(logspath, name=f'{get_config_name(config)}_')
    logging_path = os.path.join(logspath, folder_name)
    os.mkdir(logging_path)
    print(f'{logging_path = }')

    if metrics_name is None:
        metrics_name = list(filter(lambda x: config.metrics[x], config.metrics))
    # create train_log.csv where save the metrics
    with open(os.path.join(logging_path, train_log_name), 'w') as f:
        first_line = 'step,' + config.learning.loss + ',val ' + config.learning.loss
        for metric in metrics_name:
            first_line += ',' + metric
            first_line += ',val ' + metric
        f.write(first_line + '\n')
    f.close()

    # copy the config
    with open(os.path.join(logging_path, 'config.yaml'), 'w') as f:
        now = datetime.now()
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        f.write("config_metadata: 'Saving time : " + date_time + "'\n")
        for line in config_to_yaml(config):
            f.write(line + '\n')
    f.close()

    return logging_path


def config_to_yaml(config: dict, space: str='') -> str:
    """
    Transforms a dictionary (config) into a yaml line sequence.

    Args:
        config (dict): The dictionary to be transformed into yaml.
        space (str): The indentation space for each level of the yaml.

    Returns:
        str: The yaml line sequence representing the input dictionary.
    """
    intent = ' ' * 4
    config_str = []
    for key, value in config.items():
        if type(value) == EasyDict:
            if len(space) == 0:
                config_str.append('')
                config_str.append(space + '# ' + key + ' options')
            config_str.append(space + key + ':')
            config_str += config_to_yaml(value, space=space + intent)
        elif type(value) == str:
            config_str.append(space + key + ": '" + str(value) + "'")
        elif value is None:
            config_str.append(space + key + ": null")
        elif type(value) == bool:
            config_str.append(space + key + ": " + str(value).lower())
        else:
            config_str.append(space + key + ": " + str(value))
    return config_str


def train_step_logger(path: str,
                      epoch: int, 
                      train_loss: float,
                      val_loss: float, 
                      train_metrics: list[float] = [], 
                      val_metrics: list[float] = [],
                      train_log_name: str = 'train_log.csv'
                      ) -> None:
    """
    Writes loss and metrics values in the train_log.csv.

    Args:
        path (str): The path to the directory where the train_log.csv file is located.
        epoch (int): The current epoch number.
        train_loss (float): The training loss value.
        val_loss (float): The validation loss value.
        train_metrics (list[float], optional): The list of training metrics values. Defaults to an empty list.
        val_metrics (list[float], optional): The list of validation metrics values. Defaults to an empty list.
        train_log_name (str, optional): The name of the train_log.csv file. Defaults to 'train_log.csv'.
    """
    with open(os.path.join(path, train_log_name), 'a', encoding='utf8') as file:
        line = str(epoch) + ',' + str(train_loss) + ',' + str(val_loss)
        for i in range(len(train_metrics)):
            line += ',' + str(train_metrics[i])
            line += ',' + str(val_metrics[i])
        file.write(line + '\n')
    file.close()


def test_logger(path: str,
                metrics: list[str],
                values: list[float],
                dst_test_name: str = 'test_log.txt'
                ) -> None:
    """
    Creates a file 'test_log.txt' in the specified path and writes the metrics and values to it.

    Args:
        path (str): The path where the log file will be created.
        metrics (list[str]): A list of metric names.
        values (list[float]): A list of corresponding metric values.
        dst_test_name (str, optional): The name of the log file. Defaults to 'test_log.txt'.
    """
    with open(os.path.join(path, dst_test_name), 'a', encoding='utf8') as f:
        for i in range(len(metrics)):
            f.write(metrics[i] + ': ' + str(values[i]) + '\n')


def load_config(path: str='config/config.yaml') -> EasyDict:
    """
    Load a yaml into an EasyDict.

    Args:
        path (str): The path to the yaml file. Defaults to 'config/config.yaml'.

    Raises:
        FileNotFoundError: If the specified file path does not exist.

    Returns:
        EasyDict: An EasyDict object containing the loaded yaml data.
    """
    stream = open(path, 'r')
    return EasyDict(yaml.safe_load(stream))


def find_config(experiment_path: str) -> str:
    """
    Find the .yaml file in a folder and return the .yaml path.

    Args:
        experiment_path (str): The path of the folder to search for the .yaml file.

    Raises:
        FileNotFoundError: If no config.yaml file is found in the specified folder.
        FileNotFoundError: If multiple .yaml files are found in the specified folder.

    Returns:
        str: The path of the found .yaml file.

    """
    yaml_in_path = list(filter(lambda x: x[-5:] == '.yaml',
                               os.listdir(experiment_path)))

    if len(yaml_in_path) == 1:
        return os.path.join(experiment_path, yaml_in_path[0])

    if len(yaml_in_path) == 0:
        raise FileNotFoundError("ERROR: config.yaml wasn't found in", experiment_path)
    
    if len(yaml_in_path) > 0:
        raise FileNotFoundError("ERROR: multiple .yaml files were found in", experiment_path)


if __name__ == '__main__':
    #load easydict
    logging_path = 'logs/resnet_0'
    config = EasyDict(yaml.safe_load(open(os.path.join(logging_path, 'config.yaml'))))
    #print("config:", config)

    