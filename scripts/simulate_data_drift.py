import os
import sys
import json

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.failure_simulator import simulate_data_drift

def main():
    print("=" * 70)
    print("[SCENARIO 2] PRODUCTION DATA DRIFT SIMULATION & MONITORING")
    print("=" * 70)

    print("\n[Step 1] Injecting feature distribution shifts (Age +25y, Limit drop, Bill inflation)...")
    report = simulate_data_drift(n_samples=2000)

    print("\n[Step 2] Statistical Drift Detection Results:")
    print(f"* Dataset Drift Detected: {'[ALERT] YES' if report['dataset_drift_detected'] else '[OK] NO'}")
    print(f"* Drift Share: {report['drift_share']*100:.1f}% (Threshold: {report['drift_threshold']*100:.1f}%)")
    print(f"* Drifted Features: {report['drifted_features_count']} of {report['total_features_checked']}")
    print(f"* Key Drifted Columns: {', '.join(report['drifted_features'][:6])}")

    print("\n[Step 3] Diagnosis:")
    print("FAILURE TYPE: DATA_DRIFT_DETECTED")
    print("SEVERITY: HIGH")
    print("AFFECTED COMPONENT: FEATURE_DISTRIBUTIONS")
    print("RECOMMENDED ACTION: Trigger Airflow DAG / Retraining workflow on refreshed distribution.")

    print("\n[Step 4] Artifacts Generated:")
    print("* HTML Report: reports/drift_report.html")
    print("* JSON Metrics: reports/drift_metrics.json")
    print("\n[SUCCESS] DATA DRIFT SIMULATION AND MONITORING ALERT COMPLETE!")

if __name__ == "__main__":
    main()
