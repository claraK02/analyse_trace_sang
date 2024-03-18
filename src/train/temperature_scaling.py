
import torch
from easydict import EasyDict
from os.path import dirname as up

#add two folders up to the path
import os
import sys
sys.path.append(up(up(up(os.path.abspath(__file__)))))
from src.dataloader.dataloader import create_dataloader
from src.model import finetune_resnet
from utils import utils
from config.utils import load_config
import os


import matplotlib.pyplot as plt

def optimize_temperature(val_generator, model, device,config):
    """Optimize temperature on validation set"""
    nll_criterion = torch.nn.CrossEntropyLoss()
    temperature = torch.nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
    val_generator = create_dataloader(config=config, mode='val')
    print(f"Found {len(val_generator)} validation batches")

    nll_values = []  # to store NLL values

    def eval():
        total_nll = 0.0
        total_num = 0

        for i, item in enumerate(val_generator):
            x = item['image'].to(device)
            y_true = item['label'].to(device)

            logits = model.forward(x)
            y_pred = torch.nn.functional.log_softmax(logits / temperature, dim=-1)

            print("y_pred",y_pred,y_pred.shape)
            print("y_true",y_true,y_true.shape)

            loss = nll_criterion(y_pred, y_true)
            total_nll += loss.item() * x.size(0)
            total_num += x.size(0)

        avg_nll = total_nll / total_num
        nll_values.append(avg_nll)  # store NLL value

        print(f"Temperature: {temperature.item():.4f}, NLL: {avg_nll:.4f}")

        return avg_nll

    optimizer.step(eval)

    # Plot NLL values
    plt.plot(nll_values)
    plt.xlabel('Iteration')
    plt.ylabel('NLL')
    plt.title('NLL optimization curve')
    plt.show()

    return temperature.item()


if __name__ == "__main__":
    # Load the model
    import yaml
    config_file = r"logs\adv_img256_0\config.yaml"

    config = EasyDict(yaml.safe_load(open('config/config.yaml'))) 
    val_generator = create_dataloader(config=config, mode='val')
    config = load_config(config_file)
    # Get device (the first GPU if available, otherwise the CPU)
    device = utils.get_device(device_config="cuda:0")
    print("config_file: ", config_file)
    config_in_log_dir = os.path.dirname(config_file)
    print("config_in_log_dir: ", config_in_log_dir)
    model = finetune_resnet.get_finetuneresnet(config) #to get the model
    weight = utils.load_weights(config_in_log_dir, device=device)
    #move weight to device
    model.load_dict_learnable_parameters(state_dict=weight, strict=True)
    model = model.to(device)

    #print("model: ", model)

    #launch the optimization
    temperature = optimize_temperature(val_generator, model, device,config)
    print(f"Optimal temperature: {temperature}")