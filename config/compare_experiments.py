import os
import yaml
import numpy as np
import pandas as pd
from typing import Literal, List

HYPERPARAMETERS: dict[str, list[str]] = {
        'image_size': ['data', 'image_size'],
        'hidden_size': ['model', 'resnet', 'hidden_size'],
        'p_dropout': ['model', 'resnet', 'p_dropout'],
        'epochs': ['learning', 'epochs'],
        'batch_size': ['learning', 'batch_size'],
        'learning_rate': ['learning', 'learning_rate'],
        'freeze': ['model', 'resnet', 'freeze_resnet']
    }

def compare_experiments(csv_output: str='compare',
                        logs_path: str='logs',
                        config_name: str= 'config.yaml',
                        hyperparameters: dict[str, list[str]] = HYPERPARAMETERS,
                        model_name: str = 'resnet',
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
    metrics_name = get_metrics_name(model_name)
    if model_name == 'adversarial':
        model_name = 'adv'

    logs = filter(lambda x: '.' not in x, os.listdir(logs_path))    # Get all the directories
    logs = map(lambda x: os.path.join(logs_path, x), logs)          # Get the full path of the directories

    if compare_on == 'test':
        logs = filter(lambda x: test_file_name in os.listdir(x), logs)

    elif compare_on == 'val':
        logs = filter(lambda log: model_name in log, logs)              # Filter the directories based on the model name
        logs = filter(lambda x: train_log_name in os.listdir(x), logs)
        metrics_name = list(map(lambda x: f'val {x}', metrics_name))

    logs = list(logs)

    all_test_results: str = 'logs,' + list_into_str(metrics_name) + ',' + \
                            list_into_str(hyperparameters.keys()) + '\n'

    for log in logs:
        print("log : ", log)
        log_name = log.split(os.sep)[-1]
        if compare_on == 'test':
            results = get_test_results(log, test_file_name=test_file_name)
        else:
            try:
                results = get_val_results(log, train_log_name=train_log_name)
            except ValueError as e:
                print(f"Skipping {log_name} due to error: {e}")
                continue


        config = get_config(log, hyperparameters, config_name)
        results_metrics = list(map(lambda metric_name: results[metric_name],
                                metrics_name))
        all_test_results += f'{log_name},' \
                          + f'{list_into_str(results_metrics, round_up=True)}' \
                          + f',{list_into_str(config)}\n'

    csv_output: str = f'{csv_output}_{compare_on}_{model_name[:3]}.csv'
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
    print(df[val_loss_key])

    if df[val_loss_key].isna().all():
        raise ValueError(f"All values in {val_loss_key} are NaN. Cannot find minimum.")

    # Filter out NaN values before finding the minimum
    df_no_nan = df.dropna(subset=[val_loss_key])
    index_line = df_no_nan[val_loss_key].idxmin()

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


def get_metrics_name(model_name: Literal['resnet', 'adversarial', 'dann', 'trex']) -> list[str]:
    """
    Get the list of metrics names based on the given model name.

    Args:
        model_name (Literal['resnet', 'adversarial', 'dann', 'trex']): The name of the model.

    Raises:
        ValueError: If the model name is not 'resnet' or 'adversarial' or 'dann' or 'trex'.

    Returns:
        list[str]: The list of metrics names.
    """

    if model_name == 'resnet':
        metrics_name: list[str] = ['acc micro', 'acc macro', 'f1-score macro', 'top k micro']
    elif model_name == 'adversarial':
        metrics_name: list[str] = ['crossentropy', 'res loss', 'adv loss', 'resnet_top k micro', 'resnet_acc micro', 'resnet_acc macro', 'adv_acc micro']
    elif model_name == 'dann':
        metrics_name: list[str] = ['crossentropy', 'res loss', 'adv loss', 'resnet_top k micro', 'resnet_acc micro', 'resnet_acc macro', 'adv_acc micro']
    elif model_name == 'trex':
        metrics_name: list[str] = ['acc micro', 'acc macro', 'f1-score macro', 'top k micro']
    else:
        raise ValueError(f'Expected model name in {["resnet", "adversarial", "dann", "trex"]} '
                         f'but found {model_name}')

    return metrics_name

def calculate_mean_and_std(logs_path: str, output_file: str = 'mean_std_results.txt', test_file_name: str = 'test_real_log.txt') -> None:
    """
    Calculate the mean and standard deviation of metrics from test_real_log.txt files.

    Args:
        logs_path (str): Path to the directory containing experiment subdirectories.
        output_file (str, optional): Name of the output file to save the results. Defaults to 'mean_std_results.txt'.
        test_file_name (str, optional): Name of the test log file. Defaults to 'test_real_log.txt'.
    """
    metrics_data: dict[str, List[float]] = {
        'acc micro': [],
        'acc macro': [],
        'f1-score macro': [],
        'top k micro': []
    }

    logs = filter(lambda x: '.' not in x, os.listdir(logs_path))
    logs = map(lambda x: os.path.join(logs_path, x), logs)
    logs = filter(lambda x: test_file_name in os.listdir(x), logs)
    logs = list(logs)

    for log in logs:
        results = get_test_results(log, test_file_name=test_file_name)
        for metric in metrics_data.keys():
            if metric in results:
                metrics_data[metric].append(results[metric])

    mean_std_results = "Metric,Mean,Std Dev\n"
    for metric, values in metrics_data.items():
        if values:
            mean = np.mean(values)
            std_dev = np.std(values)
            mean_std_results += f"{metric},{mean:.3f},{std_dev:.3f}\n"

    output_path = os.path.join(logs_path, output_file)
    with open(output_path, mode='w', encoding='utf8') as f:
        f.write(mean_std_results)

if __name__ == '__main__':
    # compare_experiments(compare_on='val')
    # compare_experiments(compare_on='test')
    # compare_experiments(compare_on='test',
    #                     test_file_name='test_real_log.txt',
    #                     csv_output='compare_real')
    compare_experiments(logs_path="../logs/grid_search_trex_1", compare_on="val", model_name="trex")