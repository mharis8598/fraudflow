"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    """Single credit card transaction for fraud prediction."""

    Time: float = Field(..., description="Seconds elapsed since first transaction in dataset")
    V1: float = Field(..., description="PCA component 1")
    V2: float = Field(..., description="PCA component 2")
    V3: float = Field(..., description="PCA component 3")
    V4: float = Field(..., description="PCA component 4")
    V5: float = Field(..., description="PCA component 5")
    V6: float = Field(..., description="PCA component 6")
    V7: float = Field(..., description="PCA component 7")
    V8: float = Field(..., description="PCA component 8")
    V9: float = Field(..., description="PCA component 9")
    V10: float = Field(..., description="PCA component 10")
    V11: float = Field(..., description="PCA component 11")
    V12: float = Field(..., description="PCA component 12")
    V13: float = Field(..., description="PCA component 13")
    V14: float = Field(..., description="PCA component 14")
    V15: float = Field(..., description="PCA component 15")
    V16: float = Field(..., description="PCA component 16")
    V17: float = Field(..., description="PCA component 17")
    V18: float = Field(..., description="PCA component 18")
    V19: float = Field(..., description="PCA component 19")
    V20: float = Field(..., description="PCA component 20")
    V21: float = Field(..., description="PCA component 21")
    V22: float = Field(..., description="PCA component 22")
    V23: float = Field(..., description="PCA component 23")
    V24: float = Field(..., description="PCA component 24")
    V25: float = Field(..., description="PCA component 25")
    V26: float = Field(..., description="PCA component 26")
    V27: float = Field(..., description="PCA component 27")
    V28: float = Field(..., description="PCA component 28")
    Amount: float = Field(..., ge=0, description="Transaction amount in USD")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Time": 406.0,
                    "V1": -2.312, "V2": 1.951, "V3": -1.609, "V4": 3.997,
                    "V5": -0.522, "V6": -1.426, "V7": -2.537, "V8": 1.391,
                    "V9": -2.770, "V10": -2.772, "V11": 3.202, "V12": -2.899,
                    "V13": -0.595, "V14": -4.289, "V15": 0.389, "V16": -1.140,
                    "V17": -2.830, "V18": -0.016, "V19": 0.416, "V20": 0.126,
                    "V21": 0.517, "V22": -0.035, "V23": -0.465, "V24": -0.818,
                    "V25": -0.094, "V26": 0.247, "V27": 0.083, "V28": 0.078,
                    "Amount": 239.93,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Fraud prediction result."""

    fraud_probability: float = Field(..., description="Probability of fraud (0-1)")
    is_fraud: bool = Field(..., description="Whether transaction is flagged as fraud")
    risk_level: str = Field(..., description="Risk classification: LOW/MEDIUM/HIGH/CRITICAL")
    threshold: float = Field(..., description="Classification threshold used")


class HealthResponse(BaseModel):
    """API health check response."""

    status: str
    model_loaded: bool
    model_type: str | None = None
    threshold: float | None = None
    version: str = "1.0.0"


class MetricsResponse(BaseModel):
    """API usage metrics."""

    total_predictions: int
    fraud_flagged: int
    fraud_rate: float
    avg_probability: float
