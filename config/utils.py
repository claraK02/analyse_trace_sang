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
    finds a declination of a folder name so that the name is not already taken
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
                 logspath: str = 'logs') -> str:
    """
    creates a logs folder where we can find the config in confing.yaml and
    create train_log.csv which will contain the loss and metrics values
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
    with open(os.path.join(logging_path, 'train_log.csv'), 'w') as f:
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
    transforms a dictionary (config) into a yaml line sequence
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
                      val_metrics: list[float] = []
                      ) -> None:
    """
    writes loss and metrics values in the train_log.csv
    """
    with open(os.path.join(path, 'train_log.csv'), 'a', encoding='utf8') as file:
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
    creates a file 'test_log.txt' in the path
    containing for each line: metrics[i]: values[i]
    """
    with open(os.path.join(path, dst_test_name), 'a', encoding='utf8') as f:
        for i in range(len(metrics)):
            f.write(metrics[i] + ': ' + str(values[i]) + '\n')


def load_config(path: str='config/config.yaml') -> EasyDict:
    """ Load a yaml into an EasyDict"""
    stream = open(path, 'r')
    return EasyDict(yaml.safe_load(stream))


def find_config(experiment_path: str) -> str:
    """ find the .yaml file in a folder and return the .yaml path """
    yaml_in_path = list(filter(lambda x: x[-5:] == '.yaml',
                               os.listdir(experiment_path)))

    if len(yaml_in_path) == 1:
        return os.path.join(experiment_path, yaml_in_path[0])

    if len(yaml_in_path) == 0:
        raise FileNotFoundError("ERROR: config.yaml wasn't found in", experiment_path)
    
    if len(yaml_in_path) > 0:
        raise FileNotFoundError("ERROR: a lot a .yaml was found in", experiment_path)


if __name__ == '__main__':
    #load easydict
    logging_path = 'logs/resnet_0'
    config = EasyDict(yaml.safe_load(open(os.path.join(logging_path, 'config.yaml'))))
    #print("config:", config)

    