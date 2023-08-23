# Requierments
from cryptography.fernet import Fernet
import logging as log
import re
import os
import sys
import pandas as pd
from pathlib import Path
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient
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
    input_path,
    keyvault_name,
    secret_name,
    output_path
):
    """
    Component that ingest a pd.DataFrame object with filepaths and returns a compressed rar file with decrypted data.

    Args:
        input_path (pd.DataFrame): Folder where read_data output is placed, with csv extension and format:
            {
                'paths': 'https://<storage_id>.blob.core.windows.net/<container_id>/<filename><extension>'
            }
        keyvault_name (str): Parameter for decrypting key.
        secret_name (str): Parameter for decrypting key.
        output_path (uri.folder): Folder where files are compressed in .rar extension, preserving its original name.
    """
    # Create output paths
    Path('./decrypted_data').mkdir(parents=True, exist_ok=True)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    # Read input data
    df = pd.read_csv(get_file(input_path))
    # Set up keyvault client
    credential = DefaultAzureCredential()
    credential.get_token("https://management.azure.com/.default")
    secret_client = SecretClient(vault_url=f"https://{keyvault_name}.vault.azure.net/", credential=credential)
    # Fetch secret
    key = secret_client.get_secret(secret_name).value
    if isinstance(key, str):
        key = key.encode()
    # Pick up KeyVault
    my_fernet = Fernet(key)
    # Decrypt files
    for filepath in df['paths']:
        # Parameters
        filename = filepath.split('/')[-1]
        storage_name = re.findall('^https://(.*?)\.blob', filepath)[0]
        account_url = f"https://{storage_name}.blob.core.windows.net"
        container_name = re.findall(f'^https://{storage_name}.blob.core.windows.net/(.*?)/', filepath)[0]
        # Get encrypted bytes and decrypt data
        encrypted_bytes = BlobServiceClient(account_url, credential=credential)\
                .get_blob_client(container=container_name, blob=filename)\
                .download_blob(max_concurrency=1).readall()
        decrypted_bytes = my_fernet.decrypt(encrypted_bytes)
        # Save decrypted data
        with open(f'decrypted_data/{filename}', 'wb') as fd:
            fd.write(decrypted_bytes)
    # Generate output and remove decrypted data directory
    os.system(f"rar a {os.path.join(output_path,'output.rar')} ./decrypted_data")
    os.system(f"rm -rf ./decrypted_data")

if __name__=="__main__":
    fire.Fire(main)