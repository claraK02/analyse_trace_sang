import os
import yaml
from easydict import EasyDict
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import uniform, randint

CSV_PATH="logs/hyperparameters_storage.csv"

def put_in_csv(path,config,final_val_metric):
    """
    input: -path to the csv file
           -config easydict object
           -float value of the final validation metric

    output: None
    action: add a line to the csv file with the config parameters hyperparameters and the final validation metric 
    """

    #get all usefull hyperparameters
    learning_rate=config.learning.learning_rate
    batch_size=config.learning.batch_size
    epochs=config.learning.epochs
    gamma=config.learning.gamma
    dropout=config.learning.dropout
    milestones_0=config.learning.milestones[0]
    milestones_1=config.learning.milestones[1]

   
    #if the csv file does not exist, we create it and add the header

    if not os.path.isfile(path):
        with open(path, 'w') as f:
            f.write("learning_rate,batch_size,dropout_probability,epochs,gamma,milestones_0,milestones_1,final_val_metric\n")
        f.close()
    
    #add the line to the csv file
    with open(path, 'a') as f:
        f.write(f"{learning_rate},{batch_size},{dropout},{epochs},{gamma},{milestones_0},{milestones_1},{final_val_metric}\n")




def XGBoost_optimization(path, n_samples):
    """
    input: path to the csv file with the hyperparameters and the final validation metric
           n_samples: the number of hyperparameter sets to sample
    output: the best hyperparameters
    """

    # Load the results of previous experiments
    df = pd.read_csv(path)

    # Split the data into features and target
    X = df.drop('final_val_metric', axis=1)
    y = df['final_val_metric']

    # Train a XGBoost regressor on the data
    model = XGBRegressor()
    model.fit(X, y)

    # Hyperparameter search space is uniform distribution between max and min values present for each hyperparameter in the data
    search_space = {name: uniform(loc=X[name].min(), scale=X[name].max()-X[name].min()) for name in X.columns}
    print("search_space is: ",search_space)
    

    # Sample hyperparameter sets
    samples = {name: dist.rvs(n_samples) for name, dist in search_space.items()}

    # Convert the samples to a DataFrame
    samples_df = pd.DataFrame(samples)

    # Use the model to predict the validation metric for each sample
    predictions = model.predict(samples_df)

    # Find the sample with the highest predicted validation metric
    best_sample = samples_df.iloc[np.argmax(predictions)]

    return best_sample



if __name__ == '__main__':
    config = EasyDict(yaml.safe_load(open('config/config.yaml')))  # Load config file
    #put_in_csv(CSV_PATH,config,0.5)
    best_sample = XGBoost_optimization(CSV_PATH, 100)
    print("Best hyperparameters:", best_sample)