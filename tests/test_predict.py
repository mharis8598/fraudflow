"""
Tests for the prediction module.
"""

import numpy as np
import pandas as pd
import pytest

from src.predict import FraudPredictor


class TestFraudPredictor:
    def test_loads_artifacts(self, trained_artifacts):
        predictor = FraudPredictor(model_dir=trained_artifacts["model_dir"])
        assert predictor.model is not None
        assert predictor.scaler is not None
        assert predictor.metadata is not None

    def test_threshold_property(self, trained_artifacts):
        predictor = FraudPredictor(model_dir=trained_artifacts["model_dir"])
        assert predictor.threshold == 0.5

    def test_feature_names_property(self, trained_artifacts):
        predictor = FraudPredictor(model_dir=trained_artifacts["model_dir"])
        assert len(predictor.feature_names) == 30
        assert "Amount" in predictor.feature_names
        assert "V1" in predictor.feature_names

    def test_predict_single_dict(self, trained_artifacts):
        predictor = FraudPredictor(model_dir=trained_artifacts["model_dir"])

        features = {"Time": 5000.0, "Amount": 150.0}
        for i in range(1, 29):
            features[f"V{i}"] = np.random.randn()

        result = predictor.predict(features)
        assert "fraud_probability" in result
        assert "is_fraud" in result
        assert "risk_level" in result
        assert "threshold" in result
        assert 0.0 <= result["fraud_probability"] <= 1.0
        assert isinstance(result["is_fraud"], bool)

    def test_predict_single_dataframe(self, trained_artifacts):
        predictor = FraudPredictor(model_dir=trained_artifacts["model_dir"])

        row = {"Time": 5000.0, "Amount": 150.0}
        for i in range(1, 29):
            row[f"V{i}"] = np.random.randn()

        df = pd.DataFrame([row])
        result = predictor.predict(df)
        assert isinstance(result, dict)
        assert 0.0 <= result["fraud_probability"] <= 1.0

    def test_predict_batch(self, trained_artifacts):
        predictor = FraudPredictor(model_dir=trained_artifacts["model_dir"])

        rows = []
        for _ in range(5):
            row = {"Time": 5000.0, "Amount": 100.0}
            for i in range(1, 29):
                row[f"V{i}"] = np.random.randn()
            rows.append(row)

        df = pd.DataFrame(rows)
        results = predictor.predict_batch(df)
        assert len(results) == 5
        for r in results:
            assert "fraud_probability" in r
            assert "risk_level" in r

    def test_missing_model_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Model not found"):
            FraudPredictor(model_dir=str(tmp_path / "nonexistent"))


class TestRiskLevels:
    def test_critical(self):
        assert FraudPredictor._get_risk_level(0.95) == "CRITICAL"

    def test_high(self):
        assert FraudPredictor._get_risk_level(0.6) == "HIGH"

    def test_medium(self):
        assert FraudPredictor._get_risk_level(0.4) == "MEDIUM"

    def test_low(self):
        assert FraudPredictor._get_risk_level(0.1) == "LOW"
