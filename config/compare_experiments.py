import os
import yaml
import numpy as np
import pandas as pd
from typing import Literal


HYPERPARAMETERS: dict[str, list[str]] = {
        'image_size': ['data', 'image_size'],
        'hidden_size': ['model', 'resnet', 'hidden_size'],
        'p_dropout': ['model', 'resnet', 'p_dropout'],
        'epochs': ['learning', 'epochs'],
        'batch_size': ['learning', 'batch_size'],
        'learning_rate': ['learning', 'learning_rate'],
        'freeze': ['model', 'resnet', 'freeze_resnet']
    }

METRICS_NAME: list[str] = ['acc micro', 'acc macro', 'f1-score macro', 'top k macro']


def compare_experiments(csv_output: str='compare',
                        logs_path: str='logs',
                        config_name: str= 'config.yaml',
                        hyperparameters: dict[str, list[str]] = HYPERPARAMETERS,
                        metrics_name: list[str] = METRICS_NAME,
                        compare_on: Literal['val', 'test'] = 'test',
                        test_file_name: str='test_log.txt',
                        train_log_name: str='train_log.csv'
                        ) -> None:
    """
    Compare experiments and generate a CSV file with the results.

    Args:
        csv_output (str, optional): The name of the CSV output file. Defaults to 'compare'.
        logs_path (str, optional): The path to the logs directory. Defaults to 'logs'.
        config_name (str, optional): The name of the config file. Defaults to 'config.yaml'.
        hyperparameters (dict[str, list[str]], optional): A dictionary of hyperparameters. Defaults to HYPERPARAMETERS.
        metrics_name (list[str], optional): A list of metric names. Defaults to METRICS_NAME.
        compare_on (Literal['val', 'test'], optional): The type of comparison to perform. Can be 'val' or 'test'. Defaults to 'test'.
        test_file_name (str, optional): The name of the test log file. Defaults to 'test_log.txt'.
        train_log_name (str, optional): The name of the train log file. Defaults to 'train_log.csv'.
    """
    logs = filter(lambda x: '.' not in x, os.listdir(logs_path))
    logs = map(lambda x: os.path.join(logs_path, x), logs)

    if compare_on == 'test':
        logs = filter(lambda x: test_file_name in os.listdir(x), logs)

    elif compare_on == 'val':
        logs = filter(lambda x: train_log_name in os.listdir(x), logs)
        metrics_name = list(map(lambda x: f'val {x}', metrics_name))
    
    logs = list(logs)

    all_test_results: str = 'logs,' + list_into_str(metrics_name) + ',' + \
                            list_into_str(hyperparameters.keys()) + '\n'

    for log in logs:
        log_name = log.split(os.sep)[-1]
        if compare_on == 'test':
            results = get_test_results(log, test_file_name=test_file_name)
        else:
            results = get_val_results(log, train_log_name=train_log_name)
        
        config = get_config(log, hyperparameters, config_name)
        results_metrics = list(map(lambda metric_name: results[metric_name],
                                metrics_name))
        all_test_results += f'{log_name},' \
                          + f'{list_into_str(results_metrics, round_up=True)}' \
                          + f',{list_into_str(config)}\n'
    
    csv_output: str = f'{csv_output}_{compare_on}.csv'
    csv_path = os.path.join(logs_path, csv_output)
    with open(file=csv_path, mode='w', encoding='utf8') as f:
        f.write(all_test_results)
        f.close()


def get_test_results(log_path: str,
                     test_file_name: str='test_log.txt'
                     ) -> dict[str, float]:
    """
    Read the test results from a log file and return them as a dictionary.

    Args:
        log_path (str): The path to the directory containing the log file.
        test_file_name (str, optional): The name of the log file. Defaults to 'test_log.txt'.

    Raises:
        FileNotFoundError: If the log file is not found.

    Returns:
        dict[str, float]: A dictionary containing the test results, where the keys are the metric names
        and the values are the corresponding metric values.
    """
    test_file = os.path.join(log_path, test_file_name)
    if not os.path.exists(path=test_file):
        raise FileNotFoundError(f"{test_file} wasn't found")
    
    test_results: dict[str, float] = {}
    with open(test_file, mode='r', encoding='utf8') as f:
        for line in f.readlines():
            try:
                metrics_name, metric_value = line[:-1].split(': ')
            except:
                raise ValueError(f'Can split the line{line[-1]} with ": "')
            
            test_results[metrics_name] = float(metric_value)
    
    return test_results


def get_val_results(log_path: str,
                    train_log_name: str='train_log.csv'
                    ) -> dict[str, np.float64 | bool | int | str]:
    """
    Get the validation results from a train log file.

    Args:
        log_path (str): The path to the directory containing the train log file.
        train_log_name (str, optional): The name of the train log file. Defaults to 'train_log.csv'.

    Raises:
        FileNotFoundError: If the train log file is not found.

    Returns:
        dict[str, np.float64 | bool | int | str]: A dictionary containing the validation results.
    """
    train_file = os.path.join(log_path, train_log_name)
    if not os.path.exists(path=train_file):
        raise FileNotFoundError(f"{train_file} wasn't found")
    
    df = pd.read_csv(train_file)

    # find best epoch
    val_loss_key: str = list(df.keys())[2]
    index_line = df[val_loss_key].idxmin()

    # Convert the line into a dict to return it
    output: dict[str, np.float64 | bool | int | str] = dict(df.iloc[index_line])
    return output


def get_config(log_path: str,
               hyperparameters: dict[str, list[str]],
               config_name: str='config.yaml'
               ) -> list:
    """
    Retrieve configuration values based on the provided hyperparameters.

    Args:
        log_path (str): The path to the log directory.
        hyperparameters (dict[str, list[str]]): A dictionary containing the hyperparameters to retrieve.
        config_name (str, optional): The name of the configuration file. Defaults to 'config.yaml'.

    Raises:
        FileNotFoundError: If the configuration file is not found at the specified path.

    Returns:
        list: A list of configuration values corresponding to the provided hyperparameters.
    """
    config_path = os.path.join(log_path, config_name)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'config path was not found in {config_path}')
    
    config = yaml.safe_load(open(config_path))

    output = []
    for keys in hyperparameters.values():
        value = config
        for key in keys:
            value = value[key]
        output.append(value)
    
    return output


def list_into_str(l: list, sep: str=',', round_up: bool=False) -> str:
    """
    Convert a list into a string, with elements separated by a specified separator.

    Args:
        l (list): The list to be converted into a string.
        sep (str, optional): The separator to be used between elements. Defaults to ','.
        round_up (bool, optional): Whether to round up floating-point numbers to 3 decimal places. Defaults to False.

    Returns:
        str: The converted string.
    """
    output: str = ''
    for x in l:
        if round_up and type(x) in [float, np.float64]:
            output += f'{x:.3f}{sep}'
        else:
            output += f'{x}{sep}'
    return output[:-len(sep)]


if __name__ == '__main__':
    compare_experiments(compare_on='val')
    compare_experiments(compare_on='test')
    compare_experiments(compare_on='test',
                        test_file_name='test_real_log.txt',
                        csv_output='compare_real')