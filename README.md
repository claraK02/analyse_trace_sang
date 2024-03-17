


# Installation
To run the code you need python (We use python 3.10.11) and packages that is indicate in [requirements.txt](requirements.txt). You can run the following code to install all packages in the correct versions:
```bash
pip install -r requirements.txt
```

# Data

## Create data transform
```bash
python .\data_transform\get_data_transform.py
```

### Training the Model

To train the model, you need to run the `main.py` script with the `train` mode. You can specify the path to the configuration file with the `--config_path` option. If not specified, the default configuration file at `config/config.yaml` will be used.

Here's the command to train the model:

```bash
python main.py train --config_path path_to_your_config_file
```

Exemple of command to train the model:

```bash
python main.py train --config_path config/config.yaml
```

To launch the test:
    
```bash
python main.py test --config_path config/config.yaml
```
