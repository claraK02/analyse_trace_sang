import os
import argparse

from config.config import load_config, find_config
from src import train, train_adversarial, test


def main(options: dict) -> None:

    if options['mode'] == 'train':
        config = load_config(options['config_path'])

        if config.model.name == 'resnet':
            print('train resnet')
            train.train(config)
        else:
            print('train adversarial')
            train_adversarial.train(config)
    
    if options['mode'] == 'test':
        if options['path'] is None:
            raise ValueError('Please specify the path to the experiments')
        config_path = find_config(experiment_path=options['path'])
        config = load_config(config_path)
        test.test(config=config, logging_path=options['path'])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Options
    parser.add_argument('mode', default=None, type=str,
                        choices=['train', 'test'])
    parser.add_argument('--config_path', '-c', default=os.path.join('config', 'config.yaml'),
                        type=str, help="path to config (for training)")
    parser.add_argument('--path', '-p', type=str,
                        help="experiment path (for test)")
    args = parser.parse_args()
    options = vars(args)

    main(options)