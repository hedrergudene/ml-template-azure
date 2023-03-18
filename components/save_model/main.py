# Requierments
import logging as log
import os
import sys
import json
import azureml.core
from azureml.exceptions import UserErrorException, ProjectSystemException
from pathlib import Path
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


# Main method. Fire automatically allign method arguments with parse commands from console
def main(
    model_ckpt_path,
    metadata_path,
    subscription_id,
    resource_group,
    aml_workspace_name,
    experiment_name,
    output_path
):
    # Create output paths
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # Fetch paths
    model_ckpt_path = get_file(model_ckpt_path)
    metadata_path = get_file(metadata_path)
    with open(metadata_path, 'r', encoding='utf-8') as f: metadata_dct = json.load(f)
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
        model_path=model_ckpt_path,                                             # Local file to upload and register as a model.
        model_framework=metadata_dct['framework']['name'],                      # Framework used to create the model.
        model_framework_version=metadata_dct['framework']['version'],           # Version of scikit-learn used to create the model.
        #sample_input_dataset=input_dataset,
        #sample_output_dataset=output_dataset,
        resource_configuration=azureml.core.resource_configuration.ResourceConfiguration(cpu=1, memory_in_gb=2.0, gpu=0),
        description='Model checkpoint template.',
        properties={**metadata_dct['params'], **metadata_dct['metrics'], **metadata_dct['stats']},
        tags={'author': 'Antonio Zarauz Moreno, CEO @MAIACorp'}
    )
    # Write message 200
    with open(os.path.join(output_path, 'status.json'), 'w', encoding='utf-8') as f:
        json.dump({'status':200}, f)


if __name__=="__main__":
    fire.Fire(main)