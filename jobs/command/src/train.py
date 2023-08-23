# Requierments
## Basics
import logging as log
import json
import os
import sys
from uuid import uuid4
from typing import List, Dict
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
import azureml.core
from azureml.exceptions import ProjectSystemException
## Model selection
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, mean_squared_error, mean_absolute_error
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

# Function used to compute evaluation metrics
def compute_metrics(inputs:List[np.array], targets:List[np.array], is_regression:bool):
    # Formateamos salida de modelo
    y_true = np.concatenate(targets)
    y_pred_proba = np.concatenate(inputs, axis=0)
    y_pred_logproba = np.log(y_pred_proba+1e-5)
    y_pred = np.argmax(y_pred_proba, axis=-1)
    y_pred_onehot = np.zeros(shape=(y_pred.size, y_pred.max() + 1))
    y_pred_onehot[np.arange(y_pred.size), y_pred] = 1
    # regression
    if is_regression:
        # mse
        mse = mean_squared_error(y_true, y_pred)
        # mae
        mae = mean_absolute_error(y_true, y_pred)
        return {'oof_mse': mse, 'oof_mae': mae}
    else:
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


# Main code
def main(
    subscription_id,
    resource_group,
    aml_workspace_name,
    mltable_name,
    model_name,
    experiment_name,
    n_calls,
    n_initial_points,
    seed
):
    # Create output paths
    Path('./output').mkdir(parents=True, exist_ok=True)
    Path('./output_experiment').mkdir(parents=True, exist_ok=True)

    #
    # Part I: Data ingestion and preprocessing
    #

    # Import data from data asset
    log.info(f"Fetch data from Azure Data Asset:")
    # Check if given credential can get token successfully
    credential = DefaultAzureCredential()
    credential.get_token("https://management.azure.com/.default")
    # MLClient
    ml_client = MLClient(credential, subscription_id, resource_group, aml_workspace_name)
    # Model (either classification or regression)
    try:
        model = getattr(sys.modules[__name__], model_name)
    except AttributeError:
        log.error("model_name parameter is not available.")
        raise AttributeError("model_name parameter is not available.")
    is_regression = True if model_name not in ['LGBMClassifier', 'SVC', 'RandomForestClassifier'] else False
    # Fetch previously created data asset
    lv = [x.latest_version for x in ml_client.data.list()][0]
    data_asset = ml_client.data.get(name=mltable_name, version=lv)
    target_var = data_asset.tags.get('target_var')
    if target_var is None:
        log.error("Target variable has not been specified during MLTable creation as tag. Please include this information.")
        raise ValueError("Target variable has not been specified during MLTable creation as tag. Please include this information.")
    tbl = mltable.load(f'azureml:/{data_asset.id}')
    # Convert to pd.DataFrame object
    df = tbl.to_pandas_dataframe()
    # Global statistics of data
    mu_X = np.mean(df.loc[:, ~df.columns.isin([target_var])].values, axis=0)
    sigma_X = np.std(df.loc[:, ~df.columns.isin([target_var])].values, axis=0)
    mu_y = np.mean(df[target_var].values) if is_regression else 0
    sigma_y = np.std(df[target_var].values) if is_regression else 1
    ## Convert target variable into `int` in case it only contains 0-1 values
    if df[target_var].dtype=='bool': df[target_var] = df[target_var].astype('int')
    # Define the space of hyperparameters to search
    log.info(f"Setting up hyperparameter space:")
    with open('./opt_config.json', 'r') as f:
        bayes_dct = json.load(f)
    search_space = []
    if bayes_dct.get('real') is not None:
        for name, values in bayes_dct.get('real').items():
            search_space.append(Real(values[0]['min'], values[1]['max'], name=name))
    if bayes_dct.get('integer') is not None:
        for name, values in bayes_dct.get('integer').items():
            search_space.append(Integer(values[0]['min'], values[1]['max'], name=name))
    
    #
    # Part II: Model training
    #
    
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
            inputs.append(input*std_y+mean_y)
            targets.append(target*std_y+mean_y)
        # Calculate metrics from out-of-folder strategy
        metrics_dct = compute_metrics(inputs, targets, is_regression)
        with open(f'./output_experiment/{str(uuid4())}.json', 'w') as f:
            json.dump({
                **{'params':{k:str(v) for k,v in params.items()}},
                **{'metrics':{k:str(v) for k,v in metrics_dct.items()}}
            }, f, indent=4)
        return metrics_dct['mse'] if is_regression else metrics_dct['oof_catcross']
    
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
    joblib.dump(model, os.path.join('output', 'pretrained_model.joblib'))
    # Generate metadata output
    framework_name = re.findall("\'(.*?)\'",str(getattr(sys.modules[__name__], model_name)))[0].split('.')[0]
    log.info(f"Dump metadata:")
    with open(os.path.join('output', 'metadata.json'), 'w', encoding='utf8') as f:
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


    #
    # Part III: Save model
    #

    # Load files
    with open(os.path.join('output', 'metadata.json'), 'r', encoding='utf-8') as f: metadata_dct = json.load(f)
    # Azure workspace config
    try:
        ws = azureml.core.Workspace(
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=aml_workspace_name,
        )
        log.info(f"Workspace from subscription {subscription_id} and resource group {resource_group} in {aml_workspace_name} was successfully fetched.")
    except ProjectSystemException:
        log.error(f"Workspace from subscription {subscription_id} and resource group {resource_group} in {aml_workspace_name} was not found.")
        raise ProjectSystemException(f"Workspace from subscription {subscription_id} and resource group {resource_group} in {aml_workspace_name} was not found.")
    log.info(f"Wokspace info: \n\tName: {ws.name}\n\tResource group: {ws.resource_group}\n\tLocation: {ws.location}\n\tSubscription ID: {ws.subscription_id}")
    # Register the model
    log.info(f'Register the model')
    azureml.core.Model.register(
        workspace=ws,
        model_name=experiment_name,                                             # Name of the registered model in your workspace.
        model_path=os.path.join('output', 'pretrained_model.joblib'),           # Local file to upload and register as a model.
        model_framework=metadata_dct['framework']['name'],                      # Framework used to create the model.
        model_framework_version=metadata_dct['framework']['version'],           # Version of scikit-learn used to create the model.
        #sample_input_dataset=input_dataset,
        #sample_output_dataset=output_dataset,
        resource_configuration=azureml.core.resource_configuration.ResourceConfiguration(cpu=1, memory_in_gb=2.0, gpu=0),
        description='Model checkpoint template.',
        properties={**metadata_dct['params'], **metadata_dct['metrics'], **metadata_dct['stats']},
        tags={'author': 'Antonio Zarauz Moreno, CEO @MAIACorp'}
    )

# Main framework
if __name__=="__main__":
    fire.Fire(main)