# Requierments
## Basics
import logging as log
import json
import os
import sys
from uuid import uuid4
from typing import List
from pathlib import Path
import re
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
## Azure
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
import mltable
import mlflow
## Model selection
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
## Models
import sklearn
import lightgbm
from sklearn.linear_model import ElasticNet
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
## Optimisation toolkit
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args
## Fire
import fire

# Setup logs
root = log.getLogger()
root.setLevel(log.DEBUG)
handler = log.StreamHandler(sys.stdout)
handler.setLevel(log.DEBUG)
formatter = log.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

# Auxiliar method to fetch files
def get_file(f):
    f = Path(f)
    if f.is_file():
        return f
    else:
        files = list(f.iterdir())
        if len(files) == 1:
            return files[0]
        else:
            log.error(f"More than one file was found in directory: {','.join(files)}.")
            return (f"More than one file was found in directory: {','.join(files)}.", 500)

# Function used to compute evaluation metrics
def compute_metrics(inputs:List[np.array], targets:List[np.array]):
    # Formateamos salida de modelo
    y_true = np.concatenate(targets)
    y_pred_proba = np.concatenate(inputs, axis=0)
    y_pred_logproba = np.log(y_pred_proba+1e-5)
    y_pred = np.argmax(y_pred_proba, axis=-1)
    y_pred_onehot = np.zeros(shape=(y_pred.size, y_pred.max() + 1))
    y_pred_onehot[np.arange(y_pred.size), y_pred] = 1
    # accuracy
    acc = accuracy_score(y_true, y_pred)
    # precision-recall
    pr = precision_score(y_true, y_pred, average = 'weighted')
    rc = recall_score(y_true, y_pred, average = 'weighted')
    # f1 score
    f1 = f1_score(y_true, y_pred, average = 'weighted')
    # Entropía
    catcross = -np.mean((y_pred_logproba[:,None,:] @ y_pred_onehot[:,:,None]).squeeze())
    return {'oof_accuracy': acc, 'oof_precision': pr, 'oof_recall': rc, 'oof_f1':f1, 'oof_catcross':catcross}


# Main method. Fire automatically allign method arguments with parse commands from console
def main(
    input_path,
    subscription_id,
    resource_group,
    aml_workspace_name,
    model_name,
    experiment_name,
    n_calls,
    n_initial_points,
    seed,
    output_model_path,
    output_metadata_path
):
    """Azure ML component to train a supervised learning ML
       model using bayesian hyperparameter optimisation.

    Args:
        input_path (uri_folder): Path where MLTable data asset metadata is contained in JSON format
        model_name (str): Id your experiment will be named with.
        experiment_name (str): Id your experiment will be named with.
        bayes_config (str): Serialised dictionary containing experiment configuration.
        n_calls (int): Number of iterations in bayesian hyperparameter optimisation algorithm.
        n_initial_points (int): Number of initial steps in bayesian hyperparameter optimisation algorithm.
        seed (int): Random state.
        output_model_path (uri_folder): Path where model weights are placed.
        output_metadata_path (uri_folder): Path where metadata is placed.

    Returns:
        _type_: _description_
    """
    # Create output paths
    Path(output_model_path).mkdir(parents=True, exist_ok=True)
    Path(output_metadata_path).mkdir(parents=True, exist_ok=True)
    Path('./output_experiment').mkdir(parents=True, exist_ok=True)
    # Read input data
    data = get_file(input_path)
    with open(data, 'r', encoding='utf-8') as f:
        data_asset_dct = json.load(f)
    
    # Import data from data asset
    log.info(f"Fetch data from Azure Data Asset:")
    # Check if given credential can get token successfully
    credential = DefaultAzureCredential()
    credential.get_token("https://management.azure.com/.default")
    ## MLClient
    try:
        ml_client = MLClient.from_config(credential=credential)
    except Exception as ex:
        # NOTE: Update following workspace information to contain
        #       your subscription ID, resource group name, and workspace name
        client_config = {
            "subscription_id": subscription_id,
            "resource_group": resource_group,
            "workspace_name": aml_workspace_name
        }

        # write and reload from config file
        config_path = "../.azureml/config.json"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as fo:
            fo.write(json.dumps(client_config))
        ml_client = MLClient.from_config(credential=credential, path=config_path)
    ## Model (either classification or regression)
    try:
        model = getattr(sys.modules[__name__], model_name)
    except AttributeError:
        log.error("model_name parameter is not available.")
        raise AttributeError("model_name parameter is not available.")
    is_regression = True if model_name not in ['LGBMClassifier', 'SVC', 'RandomForestClassifier'] else False
    ##Fetch previously created data asset
    data_asset = ml_client.data.get(name=data_asset_dct['name'], version=data_asset_dct['version'])
    target_var = data_asset.tags.get('target_var')
    tbl = mltable.load(f'azureml:/{data_asset.id}')
    ## Convert to pd.DataFrame object
    df = tbl.to_pandas_dataframe()
    ## Global statistics of data
    mu_X = np.mean(df.loc[:, ~df.columns.isin([target_var])].values, axis=0)
    sigma_X = np.std(df.loc[:, ~df.columns.isin([target_var])].values, axis=0)
    mu_y = np.mean(df[target_var].values) if is_regression else 0
    sigma_y = np.std(df[target_var].values) if is_regression else 1
    ## Convert target variable into `int` in case it only contains 0-1 values
    target_var = data_asset.tags['target_var']
    if df[target_var].dtype=='bool': df[target_var] = df[target_var].astype('int')
    # define the space of hyperparameters to search
    log.info(f"Setting up hyperparameter space:")
    with open('./input/opt_config.json', 'r') as f:
        bayes_dct = json.load(f)
    search_space = []
    if bayes_dct.get('real') is not None:
        for name, values in bayes_dct.get('real').items():
            search_space.append(Real(values[0]['min'], values[1]['max'], name=name))
    if bayes_dct.get('integer') is not None:
        for name, values in bayes_dct.get('integer').items():
            search_space.append(Integer(values[0]['min'], values[1]['max'], name=name))
    
    # Training
    log.info(f"Start fitter training:")
    ## Function to apply param configuration to specific run
    @use_named_args(search_space)
    def evaluate_model(**params):
        # Objeto que estratifica los datos de forma balanceada
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        inputs = []
        targets = []
        for train_idx, test_idx in skf.split(df.drop(target_var, axis=1).values, df[target_var].values):
            # Obtenemos los datos de cada partición
            X_train, y_train = df.loc[train_idx, ~df.columns.isin([target_var])].values, df.loc[train_idx, target_var].values
            X_test, target = df.loc[test_idx, ~df.columns.isin([target_var])].values, df.loc[test_idx, target_var].values
            # Normalize data based on partition
            mean_X, std_X = np.mean(X_train, axis=0, keepdims=True), np.std(X_train, axis=0, keepdims=True)
            X_train, X_test = (X_train-mean_X)/std_X, (X_test-mean_X)/std_X
            if is_regression:
                mean_y, std_y = np.mean(y_train), np.std(y_train)
            else:
                mean_y, std_y = 0, 1
            y_train, target = (y_train-mean_y)/std_y, (target-mean_y)/std_y
            # Define model
            model = getattr(sys.modules[__name__], model_name)(**params)
            # Train the model
            model.fit(X_train, y_train)
            # Compute predictions
            input = model.predict_proba(X_test)
            # Store predictions and labels
            inputs.append(input)
            targets.append(target)
        # Calculate metrics from out-of-folder strategy
        metrics_dct = compute_metrics(inputs, targets)
        with open(f'./output_experiment/{str(uuid4())}.json', 'w') as f:
            json.dump({
                **{'params':{k:str(v) for k,v in params.items()}},
                **{'metrics':{k:str(v) for k,v in metrics_dct.items()}}
            }, f, indent=4)
        return metrics_dct['oof_catcross']
    
    # Perform optimization
    # Object result contains two methods, "x_iters" with array config values, and "func_vals" with target metric scores.
    # On the other hand, "x" and "fun" return the best hyperparameter configuration.
    result = gp_minimize(
        func=evaluate_model,
        dimensions=search_space,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        random_state=seed
    )
    # Fetch best metrics
    log.info(f"Fetch best metrics from logs:")
    for elem in os.listdir('./output_experiment'):
        _, extension = os.path.splitext(elem)
        if extension!='.json':
            continue
        with open(os.path.join('./output_experiment', elem), 'r') as f:
            exp_dct = json.load(f)
        if result.fun==float(exp_dct['metrics']['oof_catcross']):
            metrics_dct = exp_dct['metrics']
            break

    # MLFlow session
    log.info(f"Log metrics, artifacts and parameters:")
    experiment_id = mlflow.set_experiment(experiment_name)
    mlflow_run = mlflow.start_run()
    run_id = mlflow_run.info.run_id
    ## Metrics
    mlflow.log_metrics({k:eval(v) for k,v in metrics_dct.items()})
    ## Plot
    plot_df = pd.DataFrame(np.concatenate([result.x_iters, result.func_vals[:,None]], axis=1), columns=([x.name for x in search_space] + ['oof_catcross']))
    fig = px.parallel_coordinates(plot_df, color="oof_catcross",
                                    color_continuous_scale=px.colors.diverging.Tealrose,
                                    color_continuous_midpoint=np.mean(result.func_vals))
    fig.write_html("./output_experiment/plot.html")
    mlflow.log_artifact("./output_experiment/plot.html")
    # End run in FINISHED status
    mlflow.end_run()

    # Train model on the entire dataset
    log.info(f"Train model on the entire dataset:")
    X = (df.loc[:, ~df.columns.isin([target_var])].values-mu_X)/sigma_X
    y = (df.loc[:, target_var].values-mu_y)/sigma_y
    model = getattr(sys.modules[__name__], model_name)(**{k: eval(v) for k,v in exp_dct['params'].items()})
    model.fit(X, y)
    # Generate model output
    log.info(f"Dump model:")
    joblib.dump(model, os.path.join(output_model_path, 'pretrained_model.joblib'))
    # Generate metadata output
    framework_name = re.findall("\'(.*?)\'",str(getattr(sys.modules[__name__], model_name)))[0].split('.')[0]
    log.info(f"Dump metadata:")
    with open(os.path.join(output_metadata_path, 'metadata.json'), 'w', encoding='utf8') as f:
        json.dump({
            **{'params':{k:eval(v) for k,v in exp_dct['params'].items()}},
            **{'metrics':{k:eval(v) for k,v in exp_dct['metrics'].items()}},
            **{
            'stats':{
            'mu_X': mu_X.tolist(),
            'sigma_X': sigma_X.tolist(),
            'mu_y': mu_y,
            'sigma_y': sigma_y
            }
            },
            **{
            'framework': {
            'name': framework_name,
            'version':getattr(sys.modules[__name__], framework_name).__version__
            }
            }
        }, f, ensure_ascii=False)


if __name__=="__main__":
    fire.Fire(main)