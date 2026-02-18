"""
Model training pipeline with MLflow experiment tracking.
Trains XGBoost classifier with hyperparameter logging and model versioning.
"""

import json
import logging
import os

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.data_processing import run_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def get_default_params() -> dict:
    """Default XGBoost hyperparameters tuned for fraud detection."""
    return {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 1,  # SMOTE already balances classes
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "eval_metric": "logloss",
        "use_label_encoder": False,
    }


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict | None = None,
) -> XGBClassifier:
    """Train XGBoost classifier.

    Args:
        X_train: Training features (post-SMOTE).
        y_train: Training labels.
        params: XGBoost hyperparameters. Uses defaults if None.

    Returns:
        Trained XGBClassifier.
    """
    if params is None:
        params = get_default_params()

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    logger.info(f"Model trained with {params['n_estimators']} estimators")
    return model


def evaluate_model(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict:
    """Evaluate model on test set with comprehensive metrics.

    Args:
        model: Trained classifier.
        X_test: Test features.
        y_test: True test labels.
        threshold: Classification threshold (tune for precision/recall tradeoff).

    Returns:
        Dictionary of evaluation metrics.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "threshold": threshold,
    }

    cm = confusion_matrix(y_test, y_pred)
    metrics["true_negatives"] = int(cm[0, 0])
    metrics["false_positives"] = int(cm[0, 1])
    metrics["false_negatives"] = int(cm[1, 0])
    metrics["true_positives"] = int(cm[1, 1])

    logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1']:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Legit', 'Fraud'])}")

    return metrics


def find_optimal_threshold(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> float:
    """Find the threshold that maximizes F1 score.

    Args:
        model: Trained classifier.
        X_test: Test features.
        y_test: True labels.

    Returns:
        Optimal threshold value.
    """
    y_proba = model.predict_proba(X_test)[:, 1]

    best_f1 = 0
    best_threshold = 0.5
    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    logger.info(f"Optimal threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")
    return best_threshold


def save_model(model: XGBClassifier, scaler, feature_names: list, threshold: float):
    """Save model artifacts locally.

    Args:
        model: Trained model.
        scaler: Fitted StandardScaler.
        feature_names: List of feature column names.
        threshold: Optimal classification threshold.
    """
    joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    metadata = {
        "feature_names": feature_names,
        "threshold": threshold,
        "model_type": "XGBClassifier",
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Model artifacts saved to {MODEL_DIR}")


def run_training(
    data_filepath: str | None = None,
    params: dict | None = None,
    experiment_name: str = "fraud-detection",
) -> dict:
    """Execute full training pipeline with MLflow tracking.

    Args:
        data_filepath: Optional path to raw data CSV.
        params: XGBoost hyperparameters.
        experiment_name: MLflow experiment name.

    Returns:
        Dictionary with model, metrics, and run info.
    """
    # Process data
    logger.info("=" * 60)
    logger.info("STARTING TRAINING PIPELINE")
    logger.info("=" * 60)

    data = run_pipeline(data_filepath)

    if params is None:
        params = get_default_params()

    # Set up MLflow
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="xgboost-fraud-detection") as run:
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("smote_applied", True)
        mlflow.log_param("train_samples", len(data["X_train"]))
        mlflow.log_param("test_samples", len(data["X_test"]))

        # Train
        model = train_model(data["X_train"], data["y_train"], params)

        # Find optimal threshold
        threshold = find_optimal_threshold(model, data["X_test"], data["y_test"])

        # Evaluate with optimal threshold
        metrics = evaluate_model(model, data["X_test"], data["y_test"], threshold)

        # Log metrics to MLflow
        mlflow.log_metrics({
            "roc_auc": metrics["roc_auc"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "threshold": metrics["threshold"],
            "true_positives": metrics["true_positives"],
            "false_positives": metrics["false_positives"],
            "true_negatives": metrics["true_negatives"],
            "false_negatives": metrics["false_negatives"],
        })

        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")

        # Save locally too
        save_model(model, data["scaler"], data["feature_names"], threshold)

        logger.info(f"MLflow Run ID: {run.info.run_id}")
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)

        return {
            "model": model,
            "metrics": metrics,
            "scaler": data["scaler"],
            "feature_names": data["feature_names"],
            "threshold": threshold,
            "run_id": run.info.run_id,
        }


if __name__ == "__main__":
    results = run_training()
    print("\nFinal Metrics:")
    for k, v in results["metrics"].items():
        print(f"  {k}: {v}")
