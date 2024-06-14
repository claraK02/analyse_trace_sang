import os
import argparse

from config.utils import load_config, find_config
from config.search import Search
from src.train import train_resnet, train_adversarial, train_trex
from src import test


MODE_IMPLEMENTED = ['train', 'test', 'random_search', 'grid_search']
MODEL_IMPLEMENTED = ['resnet', 'adversarial' , 'trex']


def main(options: dict) -> None:
    """
    Run the main program with the given options.

    Args:
        options (dict): A dictionary containing the program options.

    Raises:
        ValueError: If the mode specified in options is not implemented.
        ValueError: If the model name specified in the configuration is not implemented.
        NotImplementedError: If the model name specified in the configuration is not implemented for training.
    """

    if options['mode'] not in MODE_IMPLEMENTED:
        raise ValueError(f"Expected mode in {MODE_IMPLEMENTED} but found {options['mode']}")

    # TRAINING
    if options['mode'] == 'train':
        config = load_config(options['config_path'])

        if config.model.name not in MODEL_IMPLEMENTED:
            raise ValueError(f'Expected model name in {MODEL_IMPLEMENTED} but found {config.model.name}.')
        print(f'train {config.model.name}')

        if config.model.name == 'resnet':
            train_resnet.train(config)
        
        if config.model.name == 'adversarial':
            train_adversarial.train(config)

        if config.model.name == 'trex':
            train_trex.train(config)
    
    if options['mode'] in ['random_search', 'grid_search']:
        search = Search(config_yaml_file=options['config_path'],
                        name=options['mode'])

        num_run: int = len(search)
        if options['mode'] == 'random_search':
            num_run = min(options['num_run'], len(search))

        print(f"{options['mode']} with {num_run = } runs.")

        for n_run in range(num_run):
            print(f"\n- - - experiment n°{n_run + 1}/{num_run} - - -\n")
            config = search.get_new_config()

            if config.model.name not in MODEL_IMPLEMENTED:
                raise ValueError(f'Expected model name in {MODEL_IMPLEMENTED} but found {config.model.name}.')
            print(f'train {config.model.name}')
            
            if config.model.name == 'resnet':
                train_resnet.train(config, logspath=search.get_directory())
            elif config.model.name == 'adversarial':
                train_adversarial.train(config, logspath=search.get_directory())
            elif config.model.name == 'trex':
                train_trex.train(config, logspath=search.get_directory())
            else:
                raise NotImplementedError
        
        search.compare_experiments()
    
    # TESTING
    if options['mode'] == 'test':
        if options['path'] is None:
            raise ValueError('Please specify the path to the experiments')
        
        config = load_config(find_config(experiment_path=options['path']))
        print(f'test {config.model.name}')
        
        test.test(config=config,
                  logging_path=options['path'],
                  run_real_data=options['run_on_real_data'],
                  run_silancy_metrics=options['run_saliency_metics'])


def get_options() -> dict:
    """
    Parse command line arguments and return a dictionary of options.

    Args:
        --mode, -m: str, default=None
            Chose between train, test, random_search, grid_search, infer.

        --config_path, -c: str, default='config/config.yaml'
            Path to config file (for training).

        --num_run, -n: int, default=10
            Number of experiments for random search.

        --path, -p: str
            Experiment path (for test and infer).

        --run_on_real_data, -r: str, default='false'
            Run on the real data or not.

    Returns:
        dict: A dictionary of options.
    """
    parser = argparse.ArgumentParser()

    # Options
    parser.add_argument('--mode', '-m', default=None, type=str,
                        choices=MODE_IMPLEMENTED, help='chose between train and test')
    # For training
    parser.add_argument('--config_path', '-c', default=os.path.join('config', 'config.yaml'),
                        type=str, help="path to config (for training)")
    parser.add_argument('--num_run', '-n', default=10, type=int,
                        help='number of experiment for random search')
    
    # For testing
    parser.add_argument('--path', '-p', type=str,
                        help="experiment path (for test and infer)")
    parser.add_argument('--run_on_real_data', '-r', type=str, default='false',
                        help='run on the real data or not')
    parser.add_argument('--run_saliency_metics', '-s', type=str, default='false',
                        help='run the saliency metrics or not')
    args = parser.parse_args()
    options = vars(args)

    options['run_on_real_data'] = (options['run_on_real_data'].lower() == 'true')
    options['run_saliency_metics'] = (options['run_saliency_metics'].lower() == 'true')

    return options


if __name__ == "__main__":
    options = get_options()
    main(options)