"""
Tests for the training module.
"""

import os

import numpy as np
from xgboost import XGBClassifier

from src.train import (
    evaluate_model,
    find_optimal_threshold,
    get_default_params,
    save_model,
    train_model,
)


class TestDefaultParams:
    def test_returns_dict(self):
        params = get_default_params()
        assert isinstance(params, dict)

    def test_contains_required_keys(self):
        params = get_default_params()
        assert "n_estimators" in params
        assert "max_depth" in params
        assert "learning_rate" in params
        assert "random_state" in params

    def test_smote_weight(self):
        params = get_default_params()
        assert params["scale_pos_weight"] == 1  # SMOTE handles imbalance


class TestTrainModel:
    def test_returns_xgb_classifier(self, trained_artifacts):
        model = train_model(
            trained_artifacts["X_train"],
            trained_artifacts["y_train"],
            params={
                "n_estimators": 10,
                "max_depth": 3,
                "learning_rate": 0.1,
                "random_state": 42,
                "eval_metric": "logloss",
                "use_label_encoder": False,
            },
        )
        assert isinstance(model, XGBClassifier)

    def test_model_can_predict(self, trained_artifacts):
        model = trained_artifacts["model"]
        preds = model.predict(trained_artifacts["X_test"])
        assert len(preds) == len(trained_artifacts["X_test"])
        assert set(preds).issubset({0, 1})

    def test_model_predict_proba(self, trained_artifacts):
        model = trained_artifacts["model"]
        probas = model.predict_proba(trained_artifacts["X_test"])
        assert probas.shape[1] == 2
        assert np.all(probas >= 0) and np.all(probas <= 1)


class TestEvaluateModel:
    def test_returns_metrics_dict(self, trained_artifacts):
        metrics = evaluate_model(
            trained_artifacts["model"],
            trained_artifacts["X_test"],
            trained_artifacts["y_test"],
        )
        assert "roc_auc" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "threshold" in metrics

    def test_metrics_in_valid_range(self, trained_artifacts):
        metrics = evaluate_model(
            trained_artifacts["model"],
            trained_artifacts["X_test"],
            trained_artifacts["y_test"],
        )
        for key in ["roc_auc", "precision", "recall", "f1"]:
            assert 0.0 <= metrics[key] <= 1.0

    def test_confusion_matrix_values(self, trained_artifacts):
        metrics = evaluate_model(
            trained_artifacts["model"],
            trained_artifacts["X_test"],
            trained_artifacts["y_test"],
        )
        total = (
            metrics["true_positives"]
            + metrics["true_negatives"]
            + metrics["false_positives"]
            + metrics["false_negatives"]
        )
        assert total == len(trained_artifacts["y_test"])

    def test_custom_threshold(self, trained_artifacts):
        metrics = evaluate_model(
            trained_artifacts["model"],
            trained_artifacts["X_test"],
            trained_artifacts["y_test"],
            threshold=0.8,
        )
        assert metrics["threshold"] == 0.8


class TestOptimalThreshold:
    def test_returns_float(self, trained_artifacts):
        threshold = find_optimal_threshold(
            trained_artifacts["model"],
            trained_artifacts["X_test"],
            trained_artifacts["y_test"],
        )
        assert isinstance(threshold, float)

    def test_threshold_in_range(self, trained_artifacts):
        threshold = find_optimal_threshold(
            trained_artifacts["model"],
            trained_artifacts["X_test"],
            trained_artifacts["y_test"],
        )
        assert 0.1 <= threshold <= 0.9


class TestSaveModel:
    def test_saves_all_artifacts(self, trained_artifacts, tmp_path):
        import src.train as train_module

        original_dir = train_module.MODEL_DIR
        train_module.MODEL_DIR = str(tmp_path)

        try:
            save_model(
                trained_artifacts["model"],
                trained_artifacts["scaler"],
                trained_artifacts["feature_names"],
                threshold=0.52,
            )

            assert os.path.exists(tmp_path / "model.pkl")
            assert os.path.exists(tmp_path / "scaler.pkl")
            assert os.path.exists(tmp_path / "metadata.json")
        finally:
            train_module.MODEL_DIR = original_dir
