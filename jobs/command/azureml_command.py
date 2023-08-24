# Libraries
import yaml
import sys
import json
import logging as log
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment
import fire

# Setup logs
root = log.getLogger()
root.setLevel(log.DEBUG)
handler = log.StreamHandler(sys.stdout)
handler.setLevel(log.DEBUG)
formatter = log.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)
     

# Main method. Fire automatically allign method arguments with parse commands from console
def main(
    config_path:str='./azureml_command.yaml'
):

    # Get credential token
    log.info(f"Fetch credential token:")
    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        log.error(f"Something went wrong regarding authentication. Returned error is: {ex.message}")
        return (f"Something went wrong regarding authentication. Returned error is: {ex.message}", 500)
    
    # Fetch configuration file
    with open(config_path, 'r') as f:
        config_dct = yaml.load(f, Loader=yaml.FullLoader)
    
    ## Create opt_config.json object
    log.info(f"Create hyperparameter space file:")
    with open('./src/opt_config.json', 'w') as f:
      json.dump(config_dct['train']['bayesian_search']['params'], f)

    # Azure workspace config
    log.info(f"Get MLClient:")
    ml_client = MLClient(
        credential=credential,
        subscription_id=config_dct['azure']['subscription_id'],
        resource_group_name=config_dct['azure']['resource_group'],
        workspace_name=config_dct['azure']['aml_workspace_name'],
    )

    # Job config
    log.info(f"Define command job configuration:")
    job = command(
        inputs={
            'subscription_id' : config_dct['azure']['subscription_id'],
            'resource_group' : config_dct['azure']['resource_group'],
            'aml_workspace_name' : config_dct['azure']['aml_workspace_name'],
            'mltable_name' : config_dct['train']['mltable_name'],
            'model_name' : config_dct['train']['model_name'],
            'experiment_name' : config_dct['azure']['experiment_name'],
            'n_calls' : config_dct['train']['bayesian_search']['n_calls'],
            'n_initial_points' : config_dct['train']['bayesian_search']['n_initial_points'],
            'seed' : config_dct['train']['seed']
        },
        compute=config_dct['azure']['computing']['cpu_cluster_aml_id'],
        environment=Environment(
            image="mcr.microsoft.com/azureml/curated/sklearn-1.0:11",
            conda_file="conda.yaml",
        ),
        code="./src",
        command="python train.py --subscription_id ${{inputs.subscription_id}} --resource_group ${{inputs.resource_group}} --aml_workspace_name ${{inputs.aml_workspace_name}} --mltable_name ${{inputs.mltable_name}} --model_name ${{inputs.model_name}} --experiment_name ${{inputs.experiment_name}} --n_calls ${{inputs.n_calls}} --n_initial_points ${{inputs.n_initial_points}} --seed ${{inputs.seed}}",
        experiment_name=config_dct['azure']['experiment_name'],
        display_name=config_dct['azure']['experiment_name'],
    )
    
    # Submit the run
    log.info(f"Submit the job:")
    command_job = ml_client.jobs.create_or_update(job)


if __name__=="__main__":
    fire.Fire(main)