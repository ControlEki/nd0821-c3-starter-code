"""Test Inference on API. """

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "ml"))

import pandas as pd
import requests

from model import DATA_PATH

# API URL
api_url = "http://127.0.0.1:8000/predict"

# Get sample input data
df = pd.read_csv(str(DATA_PATH).replace("census.csv", "test_census.csv"))
df.columns = [i.strip(' ').replace('-', '_') for i in df.columns]
df.drop("salary", axis=1)

input_data = df.iloc[3].to_dict()

# POST a request to the API
response = requests.post(api_url, json=input_data)

# status code
print("Status code:", response.status_code)

# result
print(response.json())