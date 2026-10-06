import os
import json
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    log_loss
)
from src.logger import app_logger

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

def load_config(config_path: str = CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def evaluate_model_performance(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, Any]:
    """
    Computes standard and business-critical classification metrics:
    - Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix, Specificity.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    auc = 0.5
    if y_prob is not None:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = 0.5

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "specificity": round(specificity, 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }
    }

def explain_recall_significance() -> str:
    """Provides business domain rationale for prioritizing Recall in Credit Risk."""
    return (
        "In Credit Risk Management, Recall (Sensitivity) measures the model's ability to identify actual defaulters. "
        "A False Negative (approving a loan for a customer who defaults) leads to severe unrecoverable capital loss. "
        "Conversely, a False Positive (flagging a safe borrower) merely prompts manual verification or extra collateral. "
        "Therefore, high Recall combined with a balanced F1-Score and ROC-AUC is prioritized to protect the lending balance sheet."
    )

def save_comparison_reports(models_metrics: Dict[str, Dict[str, Any]], config: dict = None) -> pd.DataFrame:
    """
    Saves metrics to:
    - reports/metrics.json
    - reports/model_comparison.csv
    """
    if config is None:
        config = load_config()

    reports_dir = os.path.join(BASE_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    metrics_json_path = os.path.join(BASE_DIR, config["paths"]["metrics_report"])
    comparison_csv_path = os.path.join(BASE_DIR, config["paths"]["model_comparison_csv"])

    # Save JSON report
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(models_metrics, f, indent=2)

    # Convert to clean tabular comparison DataFrame
    table_rows = []
    for model_name, metrics in models_metrics.items():
        table_rows.append({
            "Model": model_name,
            "Accuracy": metrics.get("accuracy", 0.0),
            "Precision": metrics.get("precision", 0.0),
            "Recall": metrics.get("recall", 0.0),
            "F1-Score": metrics.get("f1", 0.0),
            "ROC-AUC": metrics.get("roc_auc", 0.0),
            "Specificity": metrics.get("specificity", 0.0)
        })

    df_comparison = pd.DataFrame(table_rows)
    df_comparison.sort_values(by="F1-Score", ascending=False, inplace=True)
    df_comparison.to_csv(comparison_csv_path, index=False)

    app_logger.info(f"Model comparison reports saved:\n- {metrics_json_path}\n- {comparison_csv_path}")
    return df_comparison

if __name__ == "__main__":
    print(explain_recall_significance())
