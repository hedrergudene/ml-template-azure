#!/bin/bash
for unit in cpu gpu
do
    az ml compute create -f ./create-$unit-cluster.yaml
done