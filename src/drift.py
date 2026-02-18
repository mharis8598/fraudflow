"""
Data drift detection using Evidently AI.
Compares training data distributions against incoming prediction data.
"""

import logging
import os
from datetime import datetime

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def detect_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    save_html: bool = True,
) -> dict:
    """Run data drift detection between reference and current data.

    Args:
        reference_data: Training data (baseline distribution).
        current_data: Recent prediction data.
        save_html: Whether to save HTML drift report.

    Returns:
        Dictionary with drift detection results.
    """
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
    except ImportError:
        logger.warning("Evidently not installed. Run: pip install evidently")
        return {"error": "evidently not installed", "drift_detected": None}

    # Create drift report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)

    # Extract results
    result = report.as_dict()
    metrics = result["metrics"][0]["result"]

    drift_summary = {
        "timestamp": datetime.now().isoformat(),
        "drift_detected": metrics.get("dataset_drift", False),
        "share_drifted_features": metrics.get("share_of_drifted_columns", 0),
        "n_drifted_features": metrics.get("number_of_drifted_columns", 0),
        "n_total_features": metrics.get("number_of_columns", 0),
    }

    if save_html:
        report_path = os.path.join(
            REPORTS_DIR,
            f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
        )
        report.save_html(report_path)
        drift_summary["report_path"] = report_path
        logger.info(f"Drift report saved to {report_path}")

    if drift_summary["drift_detected"]:
        logger.warning(
            f"DATA DRIFT DETECTED: "
            f"{drift_summary['n_drifted_features']}/{drift_summary['n_total_features']} "
            f"features drifted"
        )
    else:
        logger.info("No significant data drift detected")

    return drift_summary


def quick_stats_check(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
) -> dict:
    """Lightweight drift check using basic statistics (no Evidently needed).

    Compares means and standard deviations between reference and current data.
    Useful for quick health checks.

    Args:
        reference_data: Training data.
        current_data: Recent prediction data.

    Returns:
        Dictionary with statistical comparison.
    """
    ref_stats = reference_data.describe()
    cur_stats = current_data.describe()

    # Compare means — flag if any feature mean shifts > 2 std deviations
    ref_means = ref_stats.loc["mean"]
    ref_stds = ref_stats.loc["std"]
    cur_means = cur_stats.loc["mean"]

    z_scores = ((cur_means - ref_means) / ref_stds).abs()
    drifted = z_scores[z_scores > 2].index.tolist()

    return {
        "timestamp": datetime.now().isoformat(),
        "drifted_features": drifted,
        "n_drifted": len(drifted),
        "n_total": len(z_scores),
        "drift_detected": len(drifted) > 0,
        "max_z_score": float(z_scores.max()),
    }
