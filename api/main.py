"""
FastAPI application for serving fraud detection predictions.
Endpoints: /predict, /health, /metrics
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    HealthResponse,
    MetricsResponse,
    PredictionResponse,
    TransactionRequest,
)
from src.predict import FraudPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
predictor: FraudPredictor | None = None
prediction_log: list[dict] = []
start_time: float = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global predictor, start_time
    start_time = time.time()
    try:
        predictor = FraudPredictor()
        logger.info("Model loaded successfully")
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        predictor = None
    yield
    logger.info("Shutting down API")


app = FastAPI(
    title="FraudFlow API",
    description=(
        "Real-time credit card fraud detection API. "
        "Submit transaction features and get instant fraud risk assessment."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: TransactionRequest) -> PredictionResponse:
    """Predict fraud probability for a single transaction.

    Accepts 30 transaction features (Time, V1-V28, Amount) and returns
    a fraud probability, binary classification, and risk level.
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run training first: make train",
        )

    features = transaction.model_dump()

    try:
        result = predictor.predict(features)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Log prediction for metrics
    prediction_log.append(result)

    return PredictionResponse(**result)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Check API health and model status."""
    if predictor is None:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
        )

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_type=predictor.metadata.get("model_type", "unknown"),
        threshold=predictor.threshold,
    )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    """Get prediction statistics since last restart."""
    total = len(prediction_log)
    if total == 0:
        return MetricsResponse(
            total_predictions=0,
            fraud_flagged=0,
            fraud_rate=0.0,
            avg_probability=0.0,
        )

    fraud_count = sum(1 for p in prediction_log if p["is_fraud"])
    avg_prob = sum(p["fraud_probability"] for p in prediction_log) / total

    return MetricsResponse(
        total_predictions=total,
        fraud_flagged=fraud_count,
        fraud_rate=round(fraud_count / total, 4),
        avg_probability=round(avg_prob, 6),
    )


@app.get("/")
async def root():
    """API information."""
    uptime = round(time.time() - start_time, 1) if start_time else 0
    return {
        "name": "FraudFlow API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "uptime_seconds": uptime,
    }
