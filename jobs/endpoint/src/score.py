# Libraries
import os
import logging as log
import sys
import json
import numpy as np
import azureml.core
import joblib

# Setup logs
root = log.getLogger()
root.setLevel(log.DEBUG)
handler = log.StreamHandler(sys.stdout)
handler.setLevel(log.DEBUG)
formatter = log.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "output", "pretrained_model.joblib"
    )
    # deserialize the model file back into a sklearn model
    model = joblib.load(model_path)
    log.info("Init complete")

    global mu_X, sigma_X, mu_y, sigma_y
    metadata_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "output", "metadata.json"
    )
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata_dct = json.load(f)
    mu_X = np.array(metadata_dct['stats']['mu_X'])
    sigma_X = np.array(metadata_dct['stats']['sigma_X'])
    mu_y = np.array(metadata_dct['stats']['mu_y'])
    sigma_y = np.array(metadata_dct['stats']['sigma_y'])

def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    log.info("Request received")
    data = json.loads(raw_data)["data"]
    data = (np.array(data)-mu_X)/sigma_X
    result = model.predict_proba(data).flatten()[-1]
    result = result*sigma_y+mu_y
    log.info("Request processed")
    return result