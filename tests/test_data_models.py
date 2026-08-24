"""Test data / models."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "ml"))

import pandas as pd
import pytest

from data import process_data
from model import (
    CAT_FEATURES,
    DATA_PATH,
    compute_model_metrics,
    inference,
    load_model,
    train_model,
)


@pytest.fixture
def test_data():
    data = pd.read_csv(str(DATA_PATH).replace("census.csv", "test_census.csv"))
    data.columns = [i.strip(' ').replace('-', '_') for i in data.columns]
    return data

def test_process_data(test_data):
    # Prepare train data
    X_train, y_train, encoder, lb = process_data(
        test_data, categorical_features=CAT_FEATURES, label="salary", training=True
    )

    y_train = y_train.ravel()  # Add this line to make y_train 1-dimensional

    assert X_train.shape[0] == test_data.shape[0]
    assert y_train.shape == (test_data.shape[0],)
    assert encoder is not None
    assert lb is not None

def test_train_model(test_data):
    # Prepare train data
    X_train, y_train, _, _ = process_data(
        test_data, categorical_features=CAT_FEATURES, label="salary", training=True
    )

    model = train_model(X_train, y_train)
    assert model is not None

def test_compute_model_metrics(test_data):
    # get encoder and lb
    _, encoder, lb = load_model()

    X, y, _, _ = process_data(
        test_data, categorical_features=CAT_FEATURES, label="salary", 
        training=False, encoder=encoder, lb=lb
    )   
    # Inference
    preds = inference(X)
    precision, recall, fbeta = compute_model_metrics(y, preds)
    
    assert precision == 1, f"Precision is wrong, expected 1, got {precision=}"
    assert recall == 1, f"Recall is wrong, expected 1, got {recall=}"
    assert fbeta == 1, f"F1 score  is wrong, expected 1, got {fbeta=}"

def test_inference(test_data):
    # get encoder and lb
    _, encoder, lb = load_model()

    X, _, _, _ = process_data(
        test_data, categorical_features=CAT_FEATURES, label="salary", 
        training=False, encoder=encoder, lb=lb
    )   
    # Inference
    preds = inference(X)
    print(preds)

    assert len(preds) == 4, f"Expected 4 predictions, got {len(preds)=}"

