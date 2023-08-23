# Libraries
import yaml
import sys
import json
import logging as log
import azureml.core
from azure.identity import DefaultAzureCredential
from azureml.core import Environment, ScriptRunConfig, Experiment
from azure.core.exceptions import ProjectSystemException
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
    with open('./src/opt_config.json', 'w') as f:
      json.dump(config_dct['train']['bayesian_search']['params'], f)

    # Azure workspace config
    try:
        ws = azureml.core.Workspace(
            subscription_id=config_dct['azure']['subscription_id'],
            resource_group=config_dct['azure']['resource_group'],
            workspace_name=config_dct['azure']['aml_workspace_name'],
        )
        log.info(f"Workspace from subscription {config_dct['azure']['subscription_id']} and resource group {config_dct['azure']['resource_group']} in {config_dct['azure']['aml_workspace_name']} was successfully fetched.")
    except ProjectSystemException:
        log.error(f"Workspace from subscription {config_dct['azure']['subscription_id']} and resource group {config_dct['azure']['resource_group']} in {config_dct['azure']['aml_workspace_name']} was not found.")
        raise ProjectSystemException(f"Workspace from subscription {config_dct['azure']['subscription_id']} and resource group {config_dct['azure']['resource_group']} in {config_dct['azure']['aml_workspace_name']} was not found.")
    log.info(f"Wokspace info: \n\tName: {ws.name}\n\tResource group: {ws.resource_group}\n\tLocation: {ws.location}\n\tSubscription ID: {ws.subscription_id}")

    # Environment from Dockerfile
    env = Environment.from_dockerfile(name="docker_env", dockerfile="../setup/docker/Dockerfile")

    # Job configuration
    config = ScriptRunConfig(source_directory='./src',
                            script='train.py',
                            arguments=[
                                '--subscription_id', config_dct['azure']['subscription_id'],
                                '--resource_group', config_dct['azure']['resource_group'],
                                '--aml_workspace_name', config_dct['azure']['aml_workspace_name'],
                                '--mltable_name', config_dct['train']['mltable_name'],
                                '--model_name', config_dct['train']['model_name'],
                                '--experiment_name', config_dct['azure']['experiment_name'],
                                '--target_var', config_dct['data']['target_var'],
                                '--n_calls', config_dct['train']['bayesian_search']['n_calls'],
                                '--n_initial_points', config_dct['train']['bayesian_search']['n_initial_points'],
                                '--seed', config_dct['train']['seed']
                            ],
                            environment=env,
                            compute_target=config_dct['azure']['computing']['cpu_cluster_aml_id']
                        )

    # Create an experiment
    experiment = Experiment(workspace=ws, name=config_dct['azure']['experiment_name'])
    
    # Submit the run
    run = experiment.submit(config=config)


if __name__=="__main__":
    fire.Fire(main)