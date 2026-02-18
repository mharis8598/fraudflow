"""
Tests for FastAPI endpoints.
Uses httpx test client with mocked predictor.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    """Create a test client with mocked model."""
    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = {
        "fraud_probability": 0.85,
        "is_fraud": True,
        "threshold": 0.5,
        "risk_level": "CRITICAL",
    }
    mock_predictor.metadata = {"model_type": "XGBClassifier"}
    mock_predictor.threshold = 0.5

    with patch("api.main.predictor", mock_predictor):
        yield TestClient(app)


@pytest.fixture
def sample_transaction():
    """Sample transaction payload."""
    payload = {"Time": 406.0, "Amount": 239.93}
    for i in range(1, 29):
        payload[f"V{i}"] = -1.5 + (i * 0.1)
    return payload


class TestPredictEndpoint:
    def test_predict_returns_200(self, client, sample_transaction):
        resp = client.post("/predict", json=sample_transaction)
        assert resp.status_code == 200

    def test_predict_response_shape(self, client, sample_transaction):
        resp = client.post("/predict", json=sample_transaction)
        data = resp.json()
        assert "fraud_probability" in data
        assert "is_fraud" in data
        assert "risk_level" in data
        assert "threshold" in data

    def test_predict_missing_fields(self, client):
        resp = client.post("/predict", json={"Amount": 100})
        assert resp.status_code == 422  # Validation error

    def test_predict_negative_amount(self, client):
        payload = {"Time": 0, "Amount": -50}
        for i in range(1, 29):
            payload[f"V{i}"] = 0.0
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_shape(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_response_shape(self, client):
        data = client.get("/metrics").json()
        assert "total_predictions" in data
        assert "fraud_flagged" in data


class TestRootEndpoint:
    def test_root_returns_info(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.json()["name"] == "FraudFlow API"
