
[![License](https://img.shields.io/github/license/valentingol/LeafNothingBehind?color=999)](https://stringfixer.com/fr/MIT_license)
![PythonVersion](https://img.shields.io/badge/python-3.10%20%7E3.11-informational)
![PytorchVersion](https://img.shields.io/badge/PyTorch-2.2-blue)

---

Authors: Cléa Han, Yanis Labeyrie, Adrien Zabban, Nesrine Ghannay & Clara Karkach

-----

This project focuses on the analysis of blood traces within the scope of the work of forensic expert Philippe Esperança. The objective is to develop an artificial intelligence to assist and streamline the analysis of crime scenes involving blood.

Philippe Esperança's work on blood trace analysis has resulted in the classification of these traces into 18 distinct classes, listed in the table below. Our goal is to create a deep learning model capable of predicting the class for an image of a blood trace.

|Classes                       | Description                                                                                 |
|------------------------------|---------------------------------------------------------------------------------------------|
|1- Modèle Traces passives     | Trace de sang résultant de la chute d'une goutte formée sous l’action principale de la pesanteur |
|2- Modèle Goutte à Goutte     | Ensemble de traces de sang résultant d’un liquide gouttant dans un autre liquide, dont un au moins est du sang. |
|3- Modèle Transfert par contact | Trace de sang résultant de l'apposition d’un élément ensanglanté sur la surface étudiée.        |
|4- Modèle Transfert glissé    | Trace de sang résultant du mouvement d’un élément ensanglanté en contact avec la surface étudiée. Certains critères morphologiques permettent parfois d'orienter ce mouvement. |
|5- Modèle Altération par contact | Trace de sang résultant de l'apposition d'un élément dans une trace de sang humide préexistante sur la surface étudiée. |
|6- Modèle Altération glissée  | Trace de sang résultant de l'apposition d'un élément dans une trace de sang humide préexistante sur la surface étudiée. |
|7- Modèle d'Accumulation      | Volume de sang liquide sur une surface non poreuse ou poreuse saturée.                       |
|8- Modèle Coulée              | Trace de sang résultant de l'écoulement de sang liquide présent sur la surface étudiée sous l'action principale de la pesanteur |
|9- Modèle Chute de volume     | Ensemble de traces résultant d’un volume de sang chutant ou se déversant sous l’action majoritaire de la pesanteur |
|10- Modèle Sang Propulsé      | Ensemble de traces résultant de l'éjection de sang sous l'effet de la pression sanguine      |
|11- Modèle d'éjection         | Ensemble de projections résultant de l'action de la force centrifuge lors du mouvement d'un élément ensanglanté. |
|12- Modèle Volume impacté     | Ensemble de traces de sang résultant de l'impact d'un élément dans du sang liquide sur la surface étudiée. |
|13- Modèle Imprégnation       | Accumulation de sang liquide dans une surface poreuse.                                      |
|14- Modèle Zone d'interruption | Espace non ensanglanté au sein d'un modèle et/ou d'un ensemble continu de traces de sang    |
|15- Modèle Modèle d'impact    | Ensemble de projections résultant d'un choc entre un élément et une source de sang liquide.  |
|16- Modèle Foyer de modèle d'impact | Projections circulaires faisant partie d'un modèle d'impact et qui s'observent au niveau de la zone de convergence. |
|17- Modèle Trace gravitationnelle | Projection qui atteint la surface étudiée alors qu'elle est en trajectoire descendante, sous l'action principale de la pesanteur |
|18- Modèle Sang expiré        | Ensemble de projections résultant de sang propulsé par le flux respiratoire                |

# Installation
To run the code, you need Python (we use Python 3.10.11) and the packages indicated in [requirements.txt](requirements.txt). You can run the following command to install all packages in the correct versions:
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
python run_infer.py -d data/images_to_infer -m logs_previous_models/retrain_resnet_allw_img256_2 -o data/output -s false
```

## Use Streamlit
You can also perform an inference with Streamlit by runing this command line:
```bash
streamlit run streamlit/Blood_Stain_Classification_App.py
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
python main.py --mode test --path logs_previous_models/resnet_allw_img256_2 --run_saliency_metrics false
```

### Random search and Grid search
To conduct a random search or a grid search to find the best hyperparameters, you need to create a file named `search.yaml` in the config folder with the parameters you want to test. For example, you can test finding the best learning rates and the alpha parameter for the adversarial model. Copy the following example into `search.yaml`:

```yaml
learning:
  learning_rate: [0.01, 0.005, 0.001, 0.0001]
  adv:
    learning_rate_adversary: [0.01, 0.005, 0.001, 0.0001]
    alpha: [0.001, 0.1, 0.5, 1, 2, 5, 10, 100]
```

Then, to launch 20 training runs:

```bash
python main.py --mode random_search --num_run 20
```

If you want to test all combinations, use grid search instead of random search. This will create a directory in `logs` containing all your experiments. You will also have a summary table of the performance for each experiment in a CSV file.

# Our strategy
Our strategy is described in the [report](Rapport_de_Stage_M1_GHANNAY_KARKACH.pdf) (which is in French) that we invite you to read to better understand what we have done. Unfortunately, we had to remove the images from the report because they are confidential. You can use the table below, which provides the correspondence between the names given in this report and the names given in this repository.

|Models name in the report| Model name in this repository|
|-----|------|
|LP ResNet|resnet_img256_0|
|FT LP ResNet|retrain_resnet_img256_0|
|AWL ResNet|resnet_allw_img256_2|
|FT AWL ResNet|retrain_resnet_allw_img256_2|
|Adversarial|adv_img256_1|
|DANN|


# Informations
The code of Nesrine Ghannay and Clara Karkach, was developed as part of their M1 internship project in computer science with a specialization in artificial intelligence and machine learning. For this, they took over the work and the code initially carried out by three students of the Centrale Marseille school (Cléa Han, Yanis Labeyrie, Adrien Zabban).
The original code can be found in the main branch, while the modified code is located in the main-v2 branch. 
Previous results can be found in previous_logs, and our results are in logs.