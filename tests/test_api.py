"""Run API tests."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "ml"))
sys.path.append(str(Path(__file__).parent.parent / "api"))


import pandas as pd
import pytest
from fastapi.testclient import TestClient
from main import app

from model import DATA_PATH

client = TestClient(app)

@pytest.fixture
def test_data():
    data = pd.read_csv(str(DATA_PATH).replace("census.csv", "test_census.csv"))
    data.columns = [i.strip(' ').replace('-', '_') for i in data.columns]
    data.drop("salary", axis=1)
    return data

def test_get() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to our salary category prodediction API."}


def test_predict_gt_50k(test_data) -> None:
    input_data = test_data.iloc[0].to_dict()

    # Make prediction
    response = client.post("/predict", json=input_data)
    print(response.json())
    assert response.status_code == 200
    assert response.json() == {'prediction': '>50K'}


def test_predict_lte_50k(test_data) -> None:
    input_data = test_data.iloc[3].to_dict()

    # Make prediction
    response = client.post("/predict", json=input_data)
    print(response.json())
    assert response.status_code == 200
    assert response.json() == {'prediction': '<=50K'}
