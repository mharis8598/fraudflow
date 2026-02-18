"""
Tests for data processing pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data_processing import apply_smote, clean_data, preprocess_features, split_data


class TestCleanData:
    def test_removes_duplicates(self, sample_data):
        duped = pd.concat([sample_data, sample_data.iloc[:5]])
        cleaned = clean_data(duped)
        assert len(cleaned) == len(sample_data)

    def test_no_nulls(self, sample_data):
        sample_data.iloc[0, 0] = np.nan
        cleaned = clean_data(sample_data)
        assert cleaned.isnull().sum().sum() == 0

    def test_preserves_columns(self, sample_data):
        cleaned = clean_data(sample_data)
        assert list(cleaned.columns) == list(sample_data.columns)

    def test_binary_target(self, sample_data):
        cleaned = clean_data(sample_data)
        assert set(cleaned["Class"].unique()) == {0, 1}


class TestPreprocessFeatures:
    def test_returns_correct_shapes(self, sample_data):
        X, y, scaler = preprocess_features(sample_data)
        assert X.shape[0] == len(sample_data)
        assert X.shape[1] == sample_data.shape[1] - 1  # Minus target
        assert len(y) == len(sample_data)

    def test_scaler_is_fitted(self, sample_data):
        X, y, scaler = preprocess_features(sample_data)
        assert isinstance(scaler, StandardScaler)
        assert hasattr(scaler, "mean_")

    def test_amount_scaled(self, sample_data):
        X, y, scaler = preprocess_features(sample_data)
        # Scaled values should be approximately zero-centered
        assert abs(X["Amount"].mean()) < 0.1

    def test_target_separated(self, sample_data):
        X, y, scaler = preprocess_features(sample_data)
        assert "Class" not in X.columns
        assert y.name == "Class"


class TestSplitData:
    def test_split_sizes(self, sample_data):
        X, y, _ = preprocess_features(sample_data)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        assert len(X_train) + len(X_test) == len(X)
        assert abs(len(X_test) / len(X) - 0.2) < 0.05

    def test_stratified(self, sample_data):
        X, y, _ = preprocess_features(sample_data)
        X_train, X_test, y_train, y_test = split_data(X, y)
        # Both splits should have similar fraud rates
        train_rate = y_train.mean()
        test_rate = y_test.mean()
        assert abs(train_rate - test_rate) < 0.02


class TestSMOTE:
    def test_balances_classes(self, sample_data):
        X, y, _ = preprocess_features(sample_data)
        X_train, _, y_train, _ = split_data(X, y)
        X_sm, y_sm = apply_smote(X_train, y_train)
        # After SMOTE, classes should be balanced
        assert abs(y_sm.mean() - 0.5) < 0.01

    def test_preserves_features(self, sample_data):
        X, y, _ = preprocess_features(sample_data)
        X_train, _, y_train, _ = split_data(X, y)
        X_sm, y_sm = apply_smote(X_train, y_train)
        assert list(X_sm.columns) == list(X_train.columns)

    def test_increases_minority(self, sample_data):
        X, y, _ = preprocess_features(sample_data)
        X_train, _, y_train, _ = split_data(X, y)
        original_fraud = y_train.sum()
        X_sm, y_sm = apply_smote(X_train, y_train)
        assert y_sm.sum() > original_fraud
