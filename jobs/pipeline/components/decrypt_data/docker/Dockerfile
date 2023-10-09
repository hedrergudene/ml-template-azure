# Base image
FROM mcr.microsoft.com/azureml/curated/sklearn-1.0:11

# Install dependencies
RUN pip install 'azureml-core==1.47.0' \
                'azure-identity==1.12.0' \
                'azure-storage-blob==12.13.0' \
                'azure-keyvault-secrets==4.7.0' \
                'pandas==2.0.2' \
                'numpy==1.24.3' \
                'cryptography==39.0.0' \
                'protobuf==3.20.0' \
                'pyyaml==5.4.1' \
                'fire==0.4.0'