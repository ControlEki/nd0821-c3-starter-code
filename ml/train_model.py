"""Script to train machine learning model."""

import numpy as np

# Add the necessary imports for the starter code.
import pandas as pd
from sklearn.model_selection import train_test_split

from data import process_data
from model import (
    CAT_FEATURES,
    DATA_PATH,
    compute_model_metrics,
    inference,
    save_model,
    slice_performance,
    train_model,
)

np.random.seed(42)

# Add code to load in the data.
data = pd.read_csv(DATA_PATH)
data.columns = [i.strip(' ').replace('-', '_') for i in data.columns]

# Optional enhancement, use K-fold cross validation instead of a
# train-test split.
train, test = train_test_split(data, test_size=0.20)

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=CAT_FEATURES, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=CAT_FEATURES, label="salary", 
    training=False, encoder=encoder, lb=lb
)

# Train model.
model = train_model(X_train, y_train)

# Save model and encoders
save_model(model, encoder, lb)

# Inference
predictions = inference(X_test)

# Metrics
precision, recall, fbeta = compute_model_metrics(y_test, predictions)
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F-Score: {fbeta:.2f}")

# Slice Accuracy
slice_performance(test, "education")
