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


# Function to build MLTable object
def build_MLTable(config_dct):
    # Get columns
    cols_list = []
    for idx in config_dct['data']['input_vars'].keys():
        if config_dct['data']['input_vars'].get(idx) is not None:
            cols_list += config_dct['data']['input_vars'][idx]['values']
    # Build MLTable schema
    mltable_dct = {
        '$schema': 'https://azuremlschemas.azureedge.net/latest/MLTable.schema.json',
        'type': 'mltable',
        'paths': [
        {
          "pattern": f"wasbs://{config_dct['data']['container_name']}@{config_dct['data']['storage_account_name']}.blob.core.windows.net/{config_dct['data']['regex_pattern']}"
        }
      ],
        "transformations": [
          {
            "read_delimited": {
              "encoding": "utf8",
              "header": "all_files_same_headers",
              "delimiter": ",",
              "include_path_column": "true",
              "empty_as_string": "false"
            }
          },
          {
            "keep_columns": cols_list
          },
          {
              "convert_column_types": []
          }
        ]
    }
    ## float columns
    if config_dct['data']['input_vars'].get('float') is not None:
      for col in config_dct['data']['input_vars']['float']['values']:
        mltable_dct['transformations'][2]['convert_column_types'].append(
          {
            "columns": col,
            "column_type": "float"
          }
        )
    ## int columns
    if config_dct['data']['input_vars'].get('int') is not None:
      for col in config_dct['data']['input_vars']['int']['values']:
        mltable_dct['transformations'][2]['convert_column_types'].append(
          {
            "columns": col,
            "column_type": "int"
          }
        )
    ## datetime columns
    if config_dct['data']['input_vars'].get('datetime') is not None:
      for col in config_dct['data']['input_vars']['datetime']['values']:
        mltable_dct['transformations'][2]['convert_column_types'].append(
        {
            "columns": col,
            "column_type": {
            "datetime": {
                "formats": config_dct['data']['input_vars']['datetime']['formats']
            }
          }
        }
      )
    ## boolean columns
    if config_dct['data']['input_vars'].get('boolean') is not None:
      for col in config_dct['data']['input_vars']['boolean']['values']:
        mltable_dct['transformations'][2]['convert_column_types'].append(
         {
            "columns": col,
            "column_type": {
            "boolean": {
                "mismatch_as": config_dct['data']['input_vars']['boolean']['mismatch_as'],
                "true_values": config_dct['data']['input_vars']['boolean']['true_values'],
                "false_values": config_dct['data']['input_vars']['boolean']['false_values']
              }
            }
         }
      )
    return mltable_dct
        

# Main method. Fire automatically allign method arguments with parse commands from console
def main(
    config_path:str='./config/azureml_pipeline.yaml'
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
    with open('./components/train_ml/input/opt_config.json', 'w') as f:
      json.dump(config_dct['train']['bayesian_search']['params'], f)

    ## Create MLTable object
    #mltable_dct = build_MLTable(config_dct)
    #with open('./components/data_ckpt/input/MLTable', 'w') as f:
    #    yaml.dump(mltable_dct, f, indent=4, sort_keys=False, default_flow_style=False)

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
    data_ckpt_comp = load_component(source="./components/data_ckpt/data_ckpt.yaml")
    train_ml_comp = load_component(source="./components/train_ml/train_ml.yaml")
    save_model_comp = load_component(source=f"./components/save_model/save_model.yaml")
    
    # Define a pipeline containing the previous nodes
    @pipeline(
        default_compute=config_dct['azure']['computing']['cpu_cluster_aml_id'],
    )
    def azure_ml_pipeline():
        """End to end ML project"""
        # Read 
        data_ckpt_node = data_ckpt_comp(
            subscription_id=config_dct['azure']['subscription_id'],
            resource_group=config_dct['azure']['resource_group'],
            aml_workspace_name=config_dct['azure']['aml_workspace_name'],
            data_asset_name=config_dct['data']['data_asset_name'],
            target_var=config_dct['data']['target_var']
        )
        # Train
        train_ml_node = train_ml_comp(
            input_path=data_ckpt_node.outputs.output_path,
            subscription_id=config_dct['azure']['subscription_id'],
            resource_group=config_dct['azure']['resource_group'],
            aml_workspace_name=config_dct['azure']['aml_workspace_name'],
            model_name=config_dct['train']['model_name'],
            experiment_name=config_dct['azure']['experiment_name'],
            n_calls=config_dct['train']['bayesian_search']['n_calls'],
            n_initial_points=config_dct['train']['bayesian_search']['n_initial_points'],
            seed=config_dct['train']['seed']
        )
        # Save model
        save_model_node = save_model_comp(
            model_ckpt_path=train_ml_node.outputs.output_model_path,
            metadata_path=train_ml_node.outputs.output_metadata_path,
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