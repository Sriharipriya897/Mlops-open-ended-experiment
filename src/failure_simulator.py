import os
import yaml
import json
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression

from src.logger import app_logger, log_failure
from src.data_preprocessing import load_config, clean_column_names, generate_fallback_uci_data, prepare_data
from src.drift_detection import detect_dataset_drift
from src.evaluate import evaluate_model_performance

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

def update_config_failure_flags(flags: Dict[str, Any]) -> dict:
    """Updates the failure_simulation section in config.yaml."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if "failure_simulation" not in cfg:
        cfg["failure_simulation"] = {}

    for k, v in flags.items():
        cfg["failure_simulation"][k] = v

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)

    app_logger.info(f"Updated failure simulation config flags: {flags}")
    return cfg

def reset_all_failures() -> dict:
    """Resets all failure simulation flags to normal operating conditions."""
    return update_config_failure_flags({
        "api_failure": False,
        "high_latency": False,
        "latency_delay_seconds": 0.0,
        "data_drift": False,
        "bad_model": False
    })

def simulate_data_drift(n_samples: int = 2000) -> Dict[str, Any]:
    """
    Simulates production data drift by heavily shifting demographic and credit feature distributions:
    - Aging population (age shifted +20 years)
    - Extreme credit limit drop
    - High repayment delinquency rate
    - Massive increase in bill statement amounts
    Runs statistical drift detector and generates alert.
    """
    app_logger.info(f"Generating drifted synthetic production dataset ({n_samples} samples)...")
    base_df = generate_fallback_uci_data(n_samples=n_samples, random_seed=999)

    # Intentionally apply dramatic distribution shifts
    drifted_df = base_df.copy()
    drifted_df["age"] = np.clip(drifted_df["age"] + 25, 45, 90) # Severe demographic shift
    drifted_df["limit_bal"] = np.maximum(5000, drifted_df["limit_bal"] * 0.25) # Severe credit limit contraction
    drifted_df["bill_amt1"] = drifted_df["bill_amt1"] * 3.5 # Severe debt inflation
    drifted_df["bill_amt2"] = drifted_df["bill_amt2"] * 3.2
    drifted_df["pay_0"] = np.random.choice([2, 3, 4, 5], size=n_samples, p=[0.4, 0.3, 0.2, 0.1]) # Chronic late payments
    drifted_df["pay_2"] = np.random.choice([2, 3, 4, 5], size=n_samples, p=[0.4, 0.3, 0.2, 0.1])

    drift_report = detect_dataset_drift(drifted_df)
    return drift_report

def simulate_bad_model_training(config: dict = None) -> Dict[str, Any]:
    """
    Simulates training an inferior / corrupted challenger model:
    - Inverted labels / pure noise
    - Evaluates against quality gate
    - Verifies that Quality Gate rejects it and preserves current Champion.
    """
    if config is None:
        config = load_config()

    app_logger.info("Simulating corrupted/bad candidate model training...")
    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data(config)

    # Corrupt target labels with random noise
    np.random.seed(123)
    y_corrupt = np.random.choice([0, 1], size=len(y_train), p=[0.90, 0.10])

    # Train a crippled logistic regression model on corrupted labels
    bad_model = LogisticRegression(C=0.0001, max_iter=20)
    bad_model.fit(X_train, y_corrupt)

    y_pred = bad_model.predict(X_test)
    y_prob = bad_model.predict_proba(X_test)[:, 1] if hasattr(bad_model, "predict_proba") else None
    metrics = evaluate_model_performance(y_test, y_pred, y_prob)

    min_f1 = config["model"].get("minimum_f1", 0.45)
    min_auc = config["model"].get("minimum_roc_auc", 0.70)

    quality_passed = (metrics["f1"] >= min_f1 and metrics["roc_auc"] >= min_auc)

    result = {
        "scenario": "SIMULATE_BAD_MODEL",
        "model_name": "Crippled_Logistic_Regression",
        "metrics": metrics,
        "quality_gate_passed": quality_passed,
        "thresholds": {"min_f1": min_f1, "min_roc_auc": min_auc},
        "action_taken": "APPROVED" if quality_passed else "REJECTED_CHAMPION_PRESERVED"
    }

    if not quality_passed:
        log_failure(
            failure_type="POOR_MODEL_REJECTED",
            severity="HIGH",
            component="MODEL_QUALITY_GATE",
            details=f"Crippled challenger scored F1={metrics['f1']} (min {min_f1}), ROC-AUC={metrics['roc_auc']} (min {min_auc}). Quality gate rejected deployment.",
            current_model="champion_preserved",
            recommended_action="Challenger was rejected. Stable champion remains live in production."
        )
    return result

if __name__ == "__main__":
    print("Testing Bad Model Simulation:")
    res = simulate_bad_model_training()
    print(json.dumps(res, indent=2))
