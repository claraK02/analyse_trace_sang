import os
import argparse

from config.utils import load_config, find_config
from src.infer import infer

MODEL_IMPLEMENTED = ['resnet', 'adversarial']


def main(options: dict) -> None:    
    config = load_config(find_config(experiment_path=options['modelpath']))
    if config.model.name not in MODEL_IMPLEMENTED:
        raise ValueError(f'Expected model name in {MODEL_IMPLEMENTED} but',
                         f' found {config.model.name}.')
    
    infer(infer_images_path=None,
          infer_datapath=options['datapath'],
          logging_path=options['modelpath'],
          config=config,
          dstpath=options['dstpath'],
          filename='inference_results.csv',
          run_temperature_optimization=True,
          sep=';')


def get_and_prosses_options() -> dict:
    """
    Get and process the command line options.

    Return:
        options (dict): A dictionary containing the processed options.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelpath', '-m', type=str,
                        default=os.path.join('logs', 'retrain_resnet_allw_img256_2'),
                        help="path to the model to use for the inference. \
                            default: logs/retrain_resnet_allw_img256_2")
    parser.add_argument('--datapath', '-d', type=str,
                        help="path to the folder witch contains the images to run \
                            the inference on.")
    parser.add_argument('--dstpath', '-o', type=str,
                        help="path to the folder where the inference results will be saved. \
                            default: <datapath>/inference_results")
    parser.add_argument('--plot_saliency', '-s', type=str, default='true',
                        choices=['true', 'false'],
                        help="plot the saliency map. default: true")
    args = parser.parse_args()
    options = vars(args)

    options['plot_saliency'] = (options['plot_saliency'] == 'true')

    if options['datapath'] is None:
        raise ValueError('Please specify the path to the data')
    
    if options['dstpath'] is None:
        options['dstpath'] = os.path.join(options['datapath'], 'inference_results')
        os.makedirs(options['dstpath'], exist_ok=True)
        print('The inference results will be saved at', options['dstpath'])

    return options


if __name__ == "__main__":
    options = get_and_prosses_options()
    main(options)