# Requierments
import logging as log
import json
import os
import sys
from pathlib import Path
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
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
    subscription_id,
    resource_group,
    aml_workspace_name,
    data_asset_name,
    target_var,
    output_path
):
    # Create output paths
    Path(output_path).mkdir(parents=True, exist_ok=True)
    # Check if given credential can get token successfully
    credential = DefaultAzureCredential()
    credential.get_token("https://management.azure.com/.default")
    # MLClient
    ml_client = MLClient(credential, subscription_id, resource_group, aml_workspace_name)    
    # Path to folder containing MLTable artifact (MLTable file)
    my_data = Data(
        path=f'./input',
        type=AssetTypes.MLTABLE,
        #description=data_asset_description,
        name=data_asset_name,
        tags={'target_var': target_var}
        #version='<version>'
    )
    output = ml_client.data.create_or_update(my_data)
    # If files were found, generate output
    with open(os.path.join(output_path, 'data_asset_config.json'), 'w', encoding='utf8') as f:
        json.dump(
            {
                'name':output.name,
                'version':output.version
            },
            f,
            ensure_ascii=False
        )


if __name__=="__main__":
    fire.Fire(main)