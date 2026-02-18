"""
Shared fixtures for FraudFlow test suite.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier


@pytest.fixture
def sample_data():
    """Create a small synthetic dataset mimicking credit card data."""
    np.random.seed(42)
    n = 1000
    data = {
        "Time": np.random.uniform(0, 172792, n),
        "Amount": np.random.exponential(88, n),
        "Class": np.concatenate([np.zeros(980), np.ones(20)]).astype(int),
    }
    for i in range(1, 29):
        data[f"V{i}"] = np.random.randn(n)

    return pd.DataFrame(data)


@pytest.fixture
def feature_names():
    """Standard feature column order."""
    return ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


@pytest.fixture
def trained_artifacts(sample_data, tmp_path):
    """Train a minimal model and save artifacts to a temp directory."""
    from src.data_processing import apply_smote, preprocess_features, split_data

    X, y, scaler = preprocess_features(sample_data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_sm, y_sm = apply_smote(X_train, y_train)

    model = XGBClassifier(
        n_estimators=10,
        max_depth=3,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    model.fit(X_sm, y_sm)

    # Save artifacts
    model_dir = str(tmp_path / "models")
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(model, os.path.join(model_dir, "model.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))

    metadata = {
        "feature_names": list(X.columns),
        "threshold": 0.5,
        "model_type": "XGBClassifier",
    }
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    return {
        "model": model,
        "scaler": scaler,
        "model_dir": model_dir,
        "X_train": X_sm,
        "X_test": X_test,
        "y_train": y_sm,
        "y_test": y_test,
        "feature_names": list(X.columns),
    }
