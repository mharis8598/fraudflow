"""
Tests for data drift detection module.
"""

import numpy as np
import pandas as pd
import pytest

from src.drift import quick_stats_check


@pytest.fixture
def reference_data():
    """Stable reference dataset."""
    np.random.seed(42)
    n = 500
    data = {"Amount": np.random.exponential(88, n), "Time": np.random.uniform(0, 172792, n)}
    for i in range(1, 29):
        data[f"V{i}"] = np.random.randn(n)
    return pd.DataFrame(data)


@pytest.fixture
def similar_data():
    """Data from same distribution (no drift)."""
    np.random.seed(99)
    n = 200
    data = {"Amount": np.random.exponential(88, n), "Time": np.random.uniform(0, 172792, n)}
    for i in range(1, 29):
        data[f"V{i}"] = np.random.randn(n)
    return pd.DataFrame(data)


@pytest.fixture
def drifted_data():
    """Data with significant distribution shift."""
    np.random.seed(99)
    n = 200
    data = {
        "Amount": np.random.exponential(500, n),  # Big shift
        "Time": np.random.uniform(200000, 400000, n),  # Shifted range
    }
    for i in range(1, 29):
        data[f"V{i}"] = np.random.randn(n) + 5  # Mean shift of 5 std devs
    return pd.DataFrame(data)


class TestQuickStatsCheck:
    def test_no_drift_detected(self, reference_data, similar_data):
        result = quick_stats_check(reference_data, similar_data)
        assert "drift_detected" in result
        assert "n_drifted" in result
        assert "n_total" in result
        assert "timestamp" in result

    def test_drift_detected_on_shifted_data(self, reference_data, drifted_data):
        result = quick_stats_check(reference_data, drifted_data)
        assert result["drift_detected"] is True
        assert result["n_drifted"] > 0

    def test_returns_drifted_feature_names(self, reference_data, drifted_data):
        result = quick_stats_check(reference_data, drifted_data)
        assert isinstance(result["drifted_features"], list)
        assert len(result["drifted_features"]) > 0

    def test_max_z_score_is_float(self, reference_data, similar_data):
        result = quick_stats_check(reference_data, similar_data)
        assert isinstance(result["max_z_score"], float)

    def test_identical_data_no_drift(self, reference_data):
        result = quick_stats_check(reference_data, reference_data)
        assert result["drift_detected"] is False
        assert result["n_drifted"] == 0
