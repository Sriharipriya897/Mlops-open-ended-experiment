import os
import json
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from scipy.stats import ks_2samp, chi2_contingency
from datetime import datetime

from src.logger import app_logger, log_failure
from src.data_preprocessing import clean_column_names

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

def load_config(config_path: str = CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
    """Calculates Population Stability Index (PSI) between reference and current distribution."""
    try:
        expected = expected[~np.isnan(expected)]
        actual = actual[~np.isnan(actual)]
        if len(expected) == 0 or len(actual) == 0:
            return 0.0

        percentiles = np.linspace(0, 100, num_buckets + 1)
        breakpoints = np.percentile(expected, percentiles)
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        exp_counts = np.histogram(expected, bins=breakpoints)[0]
        act_counts = np.histogram(actual, bins=breakpoints)[0]

        exp_pct = np.maximum(exp_counts / len(expected), 1e-4)
        act_pct = np.maximum(act_counts / len(actual), 1e-4)

        psi_val = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
        return float(psi_val)
    except Exception:
        return 0.0

def detect_dataset_drift(current_df: pd.DataFrame, reference_df: pd.DataFrame = None, config: dict = None) -> Dict[str, Any]:
    """
    Evaluates statistical drift across all feature columns between reference data and current production data.
    Uses two-sample KS-test for continuous features and PSI/Chi-Square for discrete features.
    """
    if config is None:
        config = load_config()

    if reference_df is None:
        ref_path = os.path.join(BASE_DIR, config["paths"]["reference_data"])
        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"Reference dataset not found at {ref_path}. Run data preparation first.")
        reference_df = pd.read_csv(ref_path)

    current_df = clean_column_names(current_df)
    reference_df = clean_column_names(reference_df)

    p_val_threshold = config["monitoring"].get("drift_p_value_threshold", 0.05)
    drift_threshold = config["monitoring"].get("drift_threshold", 0.20)

    cat_cols = ["sex", "education", "marriage", "pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]
    common_cols = [c for c in reference_df.columns if c in current_df.columns and c != "default"]

    feature_metrics = []
    drifted_count = 0

    for col in common_cols:
        ref_series = reference_df[col].dropna().values
        curr_series = current_df[col].dropna().values

        if col in cat_cols:
            # PSI for categorical / discrete columns
            psi_score = calculate_psi(ref_series, curr_series)
            is_drifted = psi_score > 0.10
            metric_type = "PSI"
            metric_val = round(psi_score, 4)
            p_val = None
        else:
            # KS Test for continuous columns
            stat, p_val = ks_2samp(ref_series, curr_series)
            is_drifted = p_val < p_val_threshold
            metric_type = "KS-statistic"
            metric_val = round(float(stat), 4)
            p_val = round(float(p_val), 5)

        if is_drifted:
            drifted_count += 1

        feature_metrics.append({
            "feature": col,
            "metric_type": metric_type,
            "metric_value": metric_val,
            "p_value": p_val,
            "drift_detected": bool(is_drifted),
            "ref_mean": round(float(np.mean(ref_series)), 2) if len(ref_series) > 0 else 0,
            "curr_mean": round(float(np.mean(curr_series)), 2) if len(curr_series) > 0 else 0
        })

    drift_share = drifted_count / len(common_cols) if common_cols else 0.0
    dataset_drift_detected = drift_share >= drift_threshold

    result = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "dataset_drift_detected": bool(dataset_drift_detected),
        "drift_share": round(drift_share, 4),
        "drift_threshold": drift_threshold,
        "total_features_checked": len(common_cols),
        "drifted_features_count": drifted_count,
        "drifted_features": [f["feature"] for f in feature_metrics if f["drift_detected"]],
        "feature_metrics": feature_metrics
    }

    # Save reports
    save_drift_reports(result, config)

    if dataset_drift_detected:
        log_failure(
            failure_type="DATA_DRIFT_DETECTED",
            severity="HIGH",
            component="FEATURE_MONITORING_ENGINE",
            details=f"Drift share {drift_share*100:.1f}% exceeds threshold {drift_threshold*100:.1f}%. Drifted features: {result['drifted_features'][:5]}...",
            recommended_action="Trigger Airflow retraining pipeline to adapt champion model to current customer distribution."
        )
    else:
        app_logger.info(f"Drift check complete: NO significant drift detected ({drifted_count}/{len(common_cols)} features).")

    return result

def generate_drift_html_report(drift_result: dict, output_path: str):
    """Generates an HTML report for data and concept drift."""
    status_color = "#ef4444" if drift_result["dataset_drift_detected"] else "#10b981"
    status_text = "DRIFT DETECTED - RETRAINING RECOMMENDED" if drift_result["dataset_drift_detected"] else "HEALTHY - DISTRIBUTIONS STABLE"
    
    rows_html = ""
    for f in drift_result["feature_metrics"]:
        badge = '<span style="background: #fee2e2; color: #991b1b; padding: 3px 8px; border-radius: 4px; font-weight: bold;">DRIFT</span>' if f["drift_detected"] else '<span style="background: #d1fae5; color: #065f46; padding: 3px 8px; border-radius: 4px;">STABLE</span>'
        p_val_str = str(f["p_value"]) if f["p_value"] is not None else "N/A"
        rows_html += f"""
        <tr>
            <td style="padding: 10px; border-bottom: 1px solid #e2e8f0; font-weight: 500;">{f['feature']}</td>
            <td style="padding: 10px; border-bottom: 1px solid #e2e8f0;">{f['metric_type']}</td>
            <td style="padding: 10px; border-bottom: 1px solid #e2e8f0;">{f['metric_value']}</td>
            <td style="padding: 10px; border-bottom: 1px solid #e2e8f0;">{p_val_str}</td>
            <td style="padding: 10px; border-bottom: 1px solid #e2e8f0;">{f['ref_mean']}</td>
            <td style="padding: 10px; border-bottom: 1px solid #e2e8f0;">{f['curr_mean']}</td>
            <td style="padding: 10px; border-bottom: 1px solid #e2e8f0; text-align: center;">{badge}</td>
        </tr>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Credit Risk MLOps - Data Drift Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }}
        .container {{ max-width: 1100px; margin: 0 auto; background: #1e293b; padding: 32px; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }}
        .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; padding-bottom: 20px; }}
        .badge-status {{ background: {status_color}; color: white; padding: 8px 16px; border-radius: 8px; font-weight: 700; letter-spacing: 0.5px; }}
        .cards-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 24px 0; }}
        .card {{ background: #0f172a; padding: 18px; border-radius: 8px; border: 1px solid #334155; }}
        .card-title {{ font-size: 13px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }}
        .card-value {{ font-size: 24px; font-weight: 700; margin-top: 6px; color: #38bdf8; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 16px; background: #0f172a; border-radius: 8px; overflow: hidden; }}
        th {{ background: #334155; padding: 12px; text-align: left; font-size: 13px; color: #cbd5e1; text-transform: uppercase; }}
        td {{ color: #e2e8f0; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1 style="margin: 0; font-size: 24px;">Evidently & Statistical Data Drift Report</h1>
                <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 14px;">Timestamp: {drift_result['timestamp']} | Reference vs Production Ingest</p>
            </div>
            <div class="badge-status">{status_text}</div>
        </div>

        <div class="cards-grid">
            <div class="card">
                <div class="card-title">Drift Share</div>
                <div class="card-value">{(drift_result['drift_share']*100):.1f}%</div>
            </div>
            <div class="card">
                <div class="card-title">Drift Threshold</div>
                <div class="card-value">{(drift_result['drift_threshold']*100):.1f}%</div>
            </div>
            <div class="card">
                <div class="card-title">Drifted Features</div>
                <div class="card-value" style="color: {status_color};">{drift_result['drifted_features_count']} / {drift_result['total_features_checked']}</div>
            </div>
            <div class="card">
                <div class="card-title">Automated Action</div>
                <div class="card-value" style="font-size: 18px; color: #a855f7;">{'Retrain Triggered' if drift_result['dataset_drift_detected'] else 'No Action Needed'}</div>
            </div>
        </div>

        <h2 style="font-size: 18px; margin-top: 32px; color: #e2e8f0;">Feature Distribution Analysis</h2>
        <table>
            <thead>
                <tr>
                    <th>Feature</th>
                    <th>Test Method</th>
                    <th>Score / Stat</th>
                    <th>p-value</th>
                    <th>Ref Mean</th>
                    <th>Curr Mean</th>
                    <th style="text-align: center;">Status</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

def save_drift_reports(drift_result: dict, config: dict = None):
    """Saves HTML and JSON drift report artifacts."""
    if config is None:
        config = load_config()

    reports_dir = os.path.join(BASE_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    json_path = os.path.join(BASE_DIR, config["paths"]["drift_metrics_json"])
    html_path = os.path.join(BASE_DIR, config["paths"]["drift_report_html"])

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(drift_result, f, indent=2)

    generate_drift_html_report(drift_result, html_path)
    app_logger.info(f"Drift report generated:\n- HTML: {html_path}\n- JSON: {json_path}")

if __name__ == "__main__":
    ref_file = os.path.join(BASE_DIR, "data", "reference", "reference.csv")
    if os.path.exists(ref_file):
        df_ref = pd.read_csv(ref_file)
        res = detect_dataset_drift(df_ref)
        print("Drift check self-test result:", res["dataset_drift_detected"])
    else:
        print("Reference data not yet prepared. Run prepare_data first.")
