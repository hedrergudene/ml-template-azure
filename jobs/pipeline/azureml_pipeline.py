# Libraries
import yaml
import sys
import json
import logging as log
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, load_component
from azure.core.exceptions import ResourceNotFoundError
from azure.ai.ml.dsl import pipeline
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
    config_path:str='./azureml_pipeline.yaml'
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
    with open('../components/train_ml/input/opt_config.json', 'w') as f:
      json.dump(config_dct['train']['bayesian_search']['params'], f)

    # Get a handle to workspace
    ml_client = MLClient(
        credential=credential,
        subscription_id=config_dct['azure']['subscription_id'],
        resource_group_name=config_dct['azure']['resource_group'],
        workspace_name=config_dct['azure']['aml_workspace_name'],
    )
    
    # Define the cpu cluster
    try:
        cpu_cluster = ml_client.compute.get(config_dct['azure']['computing']['cpu_cluster_aml_id'])
    except ResourceNotFoundError:
        log.error(f"CPU virtual machine {config_dct['azure']['computing']['cpu_cluster_aml_id']} not found.")
        raise ResourceNotFoundError(f"CPU virtual machine {config_dct['azure']['computing']['cpu_cluster_aml_id']} not found.")

    # Fetch components
    read_data_comp = load_component(source="./components/read_data/read_data.yaml")
    decrypt_data_comp = load_component(source="./components/decrypt_data/decrypt_data.yaml")
    merge_tabular_data_comp = load_component(source="./components/merge_tabular_data/merge_tabular_data.yaml")
    train_ml_comp = load_component(source="./components/train_ml/train_ml.yaml")
    save_model_comp = load_component(source=f"./components/save_model/save_model.yaml")


    # Define a pipeline containing the previous nodes
    @pipeline(
        default_compute=config_dct['azure']['computing']['cpu_cluster_aml_id'],
    )
    def azure_ml_pipeline():
        """End to end ML project"""
        # Read 
        read_data_node = read_data_comp(
            storage_id=config_dct['data']['storage_id'],
            container_id=config_dct['data']['container_id'],
            regex_pattern=config_dct['data']['regex_pattern']
        )
        # Decrypt
        decrypt_node = decrypt_data_comp(
            input_path=read_data_node.outputs.output_path,
            keyvault_name=config_dct['azure']['keyvault']['name'],
            secret_name=config_dct['azure']['keyvault']['decrypt_secret_name']
        )

        # Merge files
        merge_tabular_data_node = merge_tabular_data_comp(
            input_path=decrypt_node.outputs.output_path,
        )

        # Train
        train_ml_node = train_ml_comp(
            input_path=merge_tabular_data_node.outputs.output_path,
            model_name=config_dct['train']['model_name'],
            experiment_name=config_dct['azure']['experiment_name'],
            target_var=config_dct['data']['target_var'],
            n_calls=config_dct['train']['bayesian_search']['n_calls'],
            n_initial_points=config_dct['train']['bayesian_search']['n_initial_points'],
            seed=config_dct['train']['seed']
        )

        # Save model
        save_model_node = save_model_comp(
            input_path=train_ml_node.outputs.output_path,
            subscription_id=config_dct['azure']['subscription_id'],
            resource_group=config_dct['azure']['resource_group'],
            aml_workspace_name=config_dct['azure']['aml_workspace_name'],
            experiment_name=config_dct['azure']['experiment_name']
        )

    
    # Create a pipeline
    pipeline_job = azure_ml_pipeline()
    pipeline_job = ml_client.jobs.create_or_update(
        pipeline_job, experiment_name=config_dct['azure']['experiment_name']
    )

if __name__=="__main__":
    fire.Fire(main)