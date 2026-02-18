"""
Streamlit monitoring dashboard for FraudFlow.
Displays model metrics, prediction history, and drift alerts.
"""

import os
import json
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="FraudFlow Dashboard",
    page_icon="🛡️",
    layout="wide",
)


def get_api_health() -> dict | None:
    """Fetch API health status."""
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        return resp.json()
    except Exception:
        return None


def get_api_metrics() -> dict | None:
    """Fetch API prediction metrics."""
    try:
        resp = requests.get(f"{API_URL}/metrics", timeout=5)
        return resp.json()
    except Exception:
        return None


def load_training_metrics() -> dict | None:
    """Load latest training metrics from model metadata."""
    metadata_path = os.path.join(os.path.dirname(__file__), "..", "models", "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            return json.load(f)
    return None


# ─── Header ───────────────────────────────────────────
st.title("🛡️ FraudFlow — Fraud Detection Monitor")
st.markdown("Real-time monitoring for the credit card fraud detection system.")

# ─── System Status ────────────────────────────────────
health = get_api_health()
api_metrics = get_api_metrics()

col1, col2, col3, col4 = st.columns(4)

with col1:
    if health and health.get("status") == "healthy":
        st.metric("API Status", "🟢 Healthy")
    else:
        st.metric("API Status", "🔴 Offline")

with col2:
    if health:
        st.metric("Model Loaded", "✅ Yes" if health.get("model_loaded") else "❌ No")
    else:
        st.metric("Model Loaded", "—")

with col3:
    if health and health.get("threshold"):
        st.metric("Threshold", f"{health['threshold']:.2f}")
    else:
        st.metric("Threshold", "—")

with col4:
    if api_metrics:
        st.metric("Total Predictions", f"{api_metrics['total_predictions']:,}")
    else:
        st.metric("Total Predictions", "—")

st.divider()

# ─── Live Metrics ─────────────────────────────────────
if api_metrics and api_metrics["total_predictions"] > 0:
    st.subheader("📊 Live Prediction Metrics")

    mcol1, mcol2, mcol3 = st.columns(3)
    with mcol1:
        st.metric("Fraud Flagged", api_metrics["fraud_flagged"])
    with mcol2:
        st.metric("Fraud Rate", f"{api_metrics['fraud_rate']:.2%}")
    with mcol3:
        st.metric("Avg Probability", f"{api_metrics['avg_probability']:.4f}")

st.divider()

# ─── Test Prediction ──────────────────────────────────
st.subheader("🧪 Test a Transaction")

with st.expander("Send a test prediction to the API"):
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=150.0)

    if st.button("Predict (Random Features)"):
        # Generate random PCA features for demo
        payload = {"Time": 50000.0, "Amount": amount}
        for i in range(1, 29):
            payload[f"V{i}"] = float(np.random.randn())

        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            result = resp.json()

            if result.get("is_fraud"):
                st.error(
                    f"🚨 **FRAUD DETECTED** — "
                    f"Probability: {result['fraud_probability']:.4f} | "
                    f"Risk: {result['risk_level']}"
                )
            else:
                st.success(
                    f"✅ **Legitimate** — "
                    f"Probability: {result['fraud_probability']:.4f} | "
                    f"Risk: {result['risk_level']}"
                )
        except Exception as e:
            st.error(f"API Error: {e}")

st.divider()

# ─── Model Info ───────────────────────────────────────
st.subheader("🤖 Model Information")
metadata = load_training_metrics()

if metadata:
    st.json(metadata)
else:
    st.info("No model metadata found. Train a model first: `make train`")

# ─── Footer ───────────────────────────────────────────
st.divider()
st.caption(
    "FraudFlow v1.0.0 — Built by Mikhail Haris | "
    "[GitHub](https://github.com/mharis8598/fraudflow)"
)
