"""Relevant modeling/inference function."""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from data import process_data

BASE_DIR = Path(__file__).parent
DATA_DIR = Path(BASE_DIR).parent / "data"
MODEL_DIR = Path(BASE_DIR).parent / "model"

DATA_PATH = DATA_DIR / "census.csv"
MODEL_PATH = MODEL_DIR / "trainedmodel.pkl"
SLICE_PATH = MODEL_DIR / "slice_output.txt"

CAT_FEATURES = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    # Initialize model
    model = RandomForestClassifier(n_estimators=100)

    # Fit model
    model.fit(X_train, y_train)

    return model


def save_model(model, encoder=None, lb=None):
    """
    Save trained model.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    model_path : str
        Trained machine learning model save path.
    Returns
    -------
    None
    """
    # Save model
    artifacts = {"model": model}
    if encoder is not None:
        artifacts["encoder"] = encoder
    if lb is not None:
        artifacts["lb"] = lb

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(artifacts, f)


def load_model():
    """
    Load saved model.

    Inputs
    ------
    None

    Returns
    -------
    model : dict
        {
            "model": RandomForestClassifier,
            "encoder": OneHotEncoder,
            "lb": LabelBinarizer
        }
    """
    # Load model
    with open(MODEL_PATH, 'rb') as f:
        artifacts = pickle.load(f)

    model = artifacts["model"]
    encoder = artifacts["encoder"] if "encoder" in artifacts else None
    lb = artifacts["lb"] if "lb" in artifacts else None

    return model, encoder, lb


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    # Load Model
    model, _, _ = load_model()

    # Make predictions
    predictions = model.predict(X)

    return predictions


def slice_performance(df, feature):
    """
    Calculate and print the performance
    of a numeric feature grouped by class.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing numeric feature columns
        and a 'class' column.

    feature : str
        The feature column to analyze.
    """
    # Load Model
    _, encoder, lb = load_model()

    with open(SLICE_PATH, "w") as f:
        print(f"Accuracy for feature: {feature}", file=f)
        for cl_ in df[feature].unique():
            df_ = df.loc[df[feature] == cl_]

            # Process the test data with the process_data function.
            X, y, _, _ = process_data(df_, categorical_features=CAT_FEATURES,
                                    label="salary", training=False, encoder=encoder, lb=lb)

            # Inference
            preds = inference(X)

            precision, recall, fbeta = compute_model_metrics(y, preds)

            print(f"\n      Class: {cl_}", file=f)
            print(f"        precision: {precision:.4f}", file=f)
            print(f"        recall: {recall:.4f}", file=f)
            print(f"        fbeta: {fbeta:.4f}", file=f)


def api_inference(inference_data):
    """ Run inferences on API.

    THis is to avaid data processinf steps in inference data.

    Inputs
    ------
    model_path : str
        Trained machine learning model save path.
    inference_data : dict | pydantic.BaseModel
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    # Load Model
    model, encoder, lb = load_model()

    # Prapare data
    inference_data = dict(inference_data) if not isinstance(inference_data, dict) else inference_data
    inference_data = pd.DataFrame([inference_data])
    
    X_test, _, _, _ = process_data(
        inference_data, categorical_features=CAT_FEATURES, label=None, 
        training=False, encoder=encoder, lb=lb
    )

    # Make predictions
    predictions = model.predict(X_test)

    # Convert the binary prediction to its corresponding label
    predictions = lb.inverse_transform(predictions)[0]

    return predictions.strip(' ')
