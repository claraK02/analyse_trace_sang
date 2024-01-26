import os
import argparse

from config.config import load_config, find_config
from src.train import train


def main(options: dict) -> None:
    
    config = load_config(options['config_path'])

    if options['mode'] == 'train':
        train(config)
    
    if options['mode'] == 'test':
        raise NotImplementedError


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