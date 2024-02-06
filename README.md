# Data Directory Structure

This directory contains the following subdirectories:


- `data/`: This directory contains the data
    - `1-Modèle Traces Passives/`: This directory contains the data for the first model
    - `2-Modèle Transfert Glissé/`: This directory contains the data for the second model
    - `3-Modèle d'impact/`: This directory contains the data for the third model
       

Please refer to the individual directories for more detailed information.

# Create data transform
```bash
python .\data_transform\get_data_transform.py
```

### Training the Model

To train the model, you need to run the `main.py` script with the `train` mode. You can specify the path to the configuration file with the `--config_path` option. If not specified, the default configuration file at `config/config.yaml` will be used.

Here's the command to train the model:

```bash
python main.py train --config_path path_to_your_config_file
