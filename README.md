[![License](https://img.shields.io/github/license/valentingol/LeafNothingBehind?color=999)](https://stringfixer.com/fr/MIT_license)
![PythonVersion](https://img.shields.io/badge/python-3.10%20%7E3.11-informational)
![PytorchVersion](https://img.shields.io/badge/PyTorch-2.2-blue)

---

This GitHub repository contains the code we developed for our final project at Centrale Marseille.

Author: Cléa Han, Yanis Labeyrie, Adrien Zabban

-----

This project focuses on the analysis of blood traces within the scope of the work of forensic expert Philippe Esperança. The objective is to develop an artificial intelligence to assist and streamline the analysis of crime scenes involving blood.

Philippe Esperança's work on blood trace analysis has resulted in the classification of these traces into 18 distinct classes, listed in the table below. Our goal is to create a deep learning model capable of predicting the class for an image of a blood trace.


|Classes	                | Classes                   |
|---------------------------|---------------------------|
|Passive Traces Models	    | Droplet Models            |
|Contact Transfer Model	    | Sliding Transfer Model    |
|Contact Alteration Model	| Sliding Alteration Model  |
|Accumulation Model         | Flow Model                |
|Volume Fall Model	        | Propelled Blood Model     |
|Ejection Model	            | Impacted Volume Model     |
|Impregnation Model	        | Interruption Zone Model   |
|Impact Model	            | Impact Focal Point Model  |
|Gravitational Trace Model  | Expired Blood Model       |

# Installation
To run the code you need python (We use python 3.10.11) and packages that is indicate in [requirements.txt](requirements.txt). You can run the following code to install all packages in the correct versions:
```bash
pip install -r requirements.txt
```

# Inference
To perform an inference, execute the code `run_infer.py` with the following arguments:

| Command | Description | Default Value |
|---------|-------------|---------------|
| -d      | Path to the folder containing the images to predict | |
| -m      | Path to the model to use | logs/retrain_resnet_allw_img256_2 |
| -o      | Path where the results will be saved (creating this folder if necessary) | Subfolder "inference_results" in the data directory |
| -s      | Option to generate saliency map (true or false) | true |

You can use `python run_infer.py -h` for this documentation. Example of code execution:
```bash
python run_infer.py -d data/images_to_infer -m logs/retrain_resnet_allw_img256_2 -o data/output -s false
```

## Use Streamlit
You can also perform an inference with streamlit by using [run_app.bat](streamlit/run_app.bat) if you use Windows, or run this commende line:
```bash
streamlit run streamlit/streamlit_app.py
```

# Training (validate and testing) new models

## Config
In the file [config.yaml](config/config.yaml), you can specify all details about your model such as data augmentation, model architecture, hyperparameters, batch size, learning rate, etc. Once your training is completed, a copy of this configuration will be saved in the `logs` directory along with the model weights, its performance metrics, and learning curves.

## Data

For the code to function properly, the data for training, validation, or testing must be organized into subfolders named respectively `train_256`, `val_256`, and `test_256`, where 256 represents the image size (change the names of the subfolders accordingly). These subfolders should be grouped within a single directory, which you will specify in the file [config.yaml](config/config.yaml) under `data.path`.

For example:
```yaml
data:                                 # data parameters
  path: data/data_labo                # path to the data
  real_data_path: data/data_real      # path to the real data
  image_size: 256                     # size of the images
```

In this example, the subfolders `train_256`, `val_256`, and `test_256` containing laboratory images and real data respectively are within the directory `data/data_labo` and `data/data_real`.

## Run the code
To perform training, validation, testing, or even launch random search, you should use the file [main.py](main.py). You can run `python main.py -h` for more information.

Here are examples of launching training with the default configuration from [config](config/config.yaml):

```bash
python main.py --mode train
```

And an example of testing the `resnet_allw_img256_2` model without measuring the saliency map metrics:

```bash
python main.py --mode test --path logs/resnet_allw_img256_2 --run_saliency_metrics false
```

### Random search et Grid search
To conduct a random search or a grid search to find the best hyperparameters, you need to create a file named `search.yaml` in the config folder with the parameters you want to test. For example, you can test finding the best learning rates and the alpha parameter for the adversarial model. Copy the following example into `search.yaml`:

```yaml
learning:
  learning_rate: [0.01, 0.005, 0.001, 0.005, 0.0001]
  adv:
    learning_rate_adversary: [0.01, 0.005, 0.001, 0.005, 0.0001]
    alpha: [0.001, 0.1, 0.5, 1, 2, 5, 10, 100]
```

Then, to launch 20 training runs:

```bash
python main.py --mode random_search --num_run 20
```

If you want to test all combinations, use grid search instead of random search. This will create a directory in `logs` containing all your experiments. You will also have a summary table of the performance for each experiment in a CSV file.

# Our strategy
Our strategy is described in the [report](report/report.pdf) (which is in French) that we invite you to read to better understand what we have done. Unfortunately, we had to remove the images from the report because they are confidential. You can use the table below, which provides the correspondence between the names given in this report and the names given in this repository.

|Models name in the report| Model name in this repositoy|
|-----|------|
|LP ResNet|resnet_img256_0|
|FT LP ResNet|retrain_resnet_img256_0|
|AWL ResNet|resnet_allw_img256_2|
|FT AWL ResNet|retrain_resnet_allw_img256_2|
|Adversarial|adv_img256_1|