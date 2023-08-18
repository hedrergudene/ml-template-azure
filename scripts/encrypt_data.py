# Requierments
import re
import logging as log
import sys
from pathlib import Path
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from cryptography.fernet import Fernet
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
    input_path,
    keyvault_name,
    secret_name,
    output_path
):
    # Create output paths
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # Read input data
    with open(input_path, 'rb') as f:
        filebytes = f.read()
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
    # Encrypt file
    encrypted_bytes = my_fernet.encrypt(filebytes)
    # Save decrypted data
    with open(output_path, 'wb') as fd:
        fd.write(encrypted_bytes)


if __name__=="__main__":
    fire.Fire(main)