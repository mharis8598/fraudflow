"""
Prediction module for serving fraud detection inferences.
Loads saved model artifacts and provides prediction functions.
"""

import json
import logging
import os

import joblib
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


class FraudPredictor:
    """Loads model artifacts and provides prediction interface."""

    def __init__(self, model_dir: str | None = None):
        self.model_dir = model_dir or MODEL_DIR
        self.model = None
        self.scaler = None
        self.metadata = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Load model, scaler, and metadata from disk."""
        model_path = os.path.join(self.model_dir, "model.pkl")
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        metadata_path = os.path.join(self.model_dir, "metadata.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run training first: make train"
            )

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        with open(metadata_path) as f:
            self.metadata = json.load(f)

        logger.info(f"Loaded model from {self.model_dir}")
        logger.info(f"Threshold: {self.metadata['threshold']}")

    @property
    def threshold(self) -> float:
        return self.metadata["threshold"]

    @property
    def feature_names(self) -> list[str]:
        return self.metadata["feature_names"]

    def predict(self, features: dict | pd.DataFrame) -> dict:
        """Make a fraud prediction for a single transaction.

        Args:
            features: Dictionary or DataFrame with transaction features.
                      Must contain: V1-V28, Amount, Time.

        Returns:
            Dictionary with fraud_probability, is_fraud, and threshold.
        """
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        else:
            df = features.copy()

        # Ensure correct column order
        df = df[self.feature_names]

        # Scale Amount and Time
        df[["Amount", "Time"]] = self.scaler.transform(df[["Amount", "Time"]])

        # Predict
        proba = self.model.predict_proba(df)[:, 1]
        is_fraud = (proba >= self.threshold).astype(int)

        results = []
        for p, f in zip(proba, is_fraud):
            results.append({
                "fraud_probability": round(float(p), 6),
                "is_fraud": bool(f),
                "threshold": self.threshold,
                "risk_level": self._get_risk_level(float(p)),
            })

        return results[0] if len(results) == 1 else results

    def predict_batch(self, df: pd.DataFrame) -> list[dict]:
        """Predict fraud for a batch of transactions.

        Args:
            df: DataFrame with multiple transactions.

        Returns:
            List of prediction dictionaries.
        """
        df = df[self.feature_names].copy()
        df[["Amount", "Time"]] = self.scaler.transform(df[["Amount", "Time"]])

        probas = self.model.predict_proba(df)[:, 1]
        predictions = (probas >= self.threshold).astype(int)

        return [
            {
                "fraud_probability": round(float(p), 6),
                "is_fraud": bool(f),
                "threshold": self.threshold,
                "risk_level": self._get_risk_level(float(p)),
            }
            for p, f in zip(probas, predictions)
        ]

    @staticmethod
    def _get_risk_level(probability: float) -> str:
        """Classify risk level based on fraud probability."""
        if probability >= 0.8:
            return "CRITICAL"
        elif probability >= 0.5:
            return "HIGH"
        elif probability >= 0.3:
            return "MEDIUM"
        else:
            return "LOW"
