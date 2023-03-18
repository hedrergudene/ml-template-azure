#!/bin/bash
az ml component create --file ./components/data_ckpt/data_ckpt.yaml --version 1 --resource-group $2 --workspace-name $1
az ml component create --file ./components/train/train.yaml --version 1 --resource-group $2 --workspace-name $1
az ml component create --file ./components/save_model/save_model.yaml --version 1 --resource-group $2 --workspace-name $1