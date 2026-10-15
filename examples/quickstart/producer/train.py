"""Train a churn classifier inside a tracked run. Run from the producer folder:

    python train.py
"""
import csv
import pickle
from pathlib import Path

import mlflow
from sklearn.linear_model import LogisticRegression

import mldna

FEATURES = ["tenure_months", "monthly_charges", "support_calls"]


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    X = [[float(row[name]) for name in FEATURES] for row in rows]
    y = [int(row["churned"]) for row in rows]
    return X, y


with mldna.track_run():
    X, y = load("dataset/customers.csv")
    model = LogisticRegression(max_iter=1000).fit(X, y)

    mlflow.log_param("max_iter", 1000)
    mlflow.log_metric("accuracy", model.score(X, y))

    # Saved inside the `with` block, so mldna records its SHA-256 on this run.
    Path("models").mkdir(exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

print(f"Trained on {len(y)} rows, accuracy {model.score(X, y):.3f}. Saved models/model.pkl")
