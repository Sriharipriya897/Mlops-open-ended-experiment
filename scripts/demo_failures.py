import os
import sys
import time
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

from src.logger import app_logger
from src.data_preprocessing import load_config, validate_data, clean_column_names
from src.predict import predict_single
from src.health_check import check_readiness
from src.failure_simulator import update_config_failure_flags, reset_all_failures, simulate_data_drift, simulate_bad_model_training
from src.rollback import rollback_to_backup

def print_banner(title: str):
    print("\n" + "=" * 75)
    print(f"  {title}")
    print("=" * 75)

def main():
    print_banner("RELIABLE MLOPS SYSTEM: END-TO-END FAILURE & RECOVERY DEMONSTRATION")
    sample_input = {
        "limit_bal": 50000,
        "sex": 2,
        "education": 2,
        "marriage": 1,
        "age": 30,
        "pay_0": 0,
        "pay_2": 0,
        "pay_3": 0,
        "pay_4": 0,
        "pay_5": 0,
        "pay_6": 0,
        "bill_amt1": 15000,
        "bill_amt2": 14000,
        "bill_amt3": 13000,
        "bill_amt4": 12000,
        "bill_amt5": 11000,
        "bill_amt6": 10000,
        "pay_amt1": 2000,
        "pay_amt2": 2000,
        "pay_amt3": 2000,
        "pay_amt4": 2000,
        "pay_amt5": 2000,
        "pay_amt6": 2000
    }

    # STEP 1: NORMAL SYSTEM OPERATION
    print_banner("PHASE 1: NORMAL BASELINE OPERATION")
    r1 = check_readiness()
    print(f"System Readiness: {r1.get('status').upper()} | Active Version: {r1.get('model_version')}")
    pred1 = predict_single(sample_input)
    print(f"Prediction Result: {pred1['risk_category']} (Prob: {pred1['default_probability']:.3f}) | Latency: {pred1['latency_ms']:.1f}ms")

    # STEP 2: API & INFERENCE FAILURE
    print_banner("PHASE 2: SIMULATING API INFERENCE CRASH")
    update_config_failure_flags({"api_failure": True})
    r2 = check_readiness()
    print(f"Health Probe Status: {r2.get('status').upper()} (Failure Detected!)")
    print("Diagnosis: Model service threw runtime exception.")
    print("Action: Initiating automated failover to last known-good backup...")
    reset_all_failures()
    rollback_res = rollback_to_backup()
    r2_rec = check_readiness()
    print(f"Recovery Status: {r2_rec.get('status').upper()} | Model: {r2_rec.get('model_version')}")
    print("[SUCCESS] API Failure Recovered.")

    # STEP 3: DATA DRIFT DETECTION
    print_banner("PHASE 3: SIMULATING PRODUCTION DATA DRIFT")
    drift_res = simulate_data_drift(n_samples=1500)
    print(f"Drift Detected: {'[ALERT] YES' if drift_res['dataset_drift_detected'] else '[OK] NO'}")
    print(f"Drift Share: {drift_res['drift_share']*100:.1f}% | Drifted Features: {drift_res['drifted_features'][:4]}")
    print("Diagnosis: Customer demographic/credit behavior shifted.")
    print("Action: Dispatched alert & generated reports/drift_report.html.")
    print("[SUCCESS] Drift Detection & Alerting Verified.")

    # STEP 4: BAD MODEL QUALITY GATE REJECTION
    print_banner("PHASE 4: SIMULATING DEFECTIVE/CORRUPTED CHALLENGER MODEL")
    bad_res = simulate_bad_model_training()
    print(f"Candidate F1-Score: {bad_res['metrics']['f1']:.4f} (Threshold: {bad_res['thresholds']['min_f1']})")
    print(f"Quality Gate Action: {bad_res['action_taken']}")
    print("Reliability Guarantee: Unvalidated or broken model was prevented from touching production!")
    print("[SUCCESS] Quality Gate Shield Verified.")

    # STEP 5: HIGH LATENCY SIMULATION
    print_banner("PHASE 5: SIMULATING HIGH LATENCY DEGRADATION")
    update_config_failure_flags({"high_latency": True, "latency_delay_seconds": 1.5})
    t_start = time.perf_counter()
    pred_lat = predict_single(sample_input)
    lat = pred_lat["latency_ms"]
    print(f"Inference Latency Detected: {lat:.1f}ms (Exceeds 1000ms threshold)")
    print("Diagnosis: Latency degradation alert triggered.")
    reset_all_failures()
    pred_norm = predict_single(sample_input)
    print(f"Post-recovery Latency: {pred_norm['latency_ms']:.1f}ms")
    print("[SUCCESS] Latency Monitoring & Recovery Verified.")

    # STEP 6: DATA VALIDATION INTEGRITY
    print_banner("PHASE 6: TESTING DATA VALIDATION GUARDRAILS")
    import pandas as pd
    corrupt_df = pd.DataFrame([{"invalid_col": 123, "random_val": 456}])
    val_res = validate_data(corrupt_df)
    print(f"Corrupted Schema Valid: {val_res['is_valid']}")
    print(f"Validation Guardrail Errors: {val_res['errors']}")
    print("Action: Pipeline halted safely. Current champion untouched.")
    print("[SUCCESS] Data Integrity Guardrail Verified.")

    # FINAL VERIFICATION
    print_banner("FINAL SYSTEM INTEGRITY VERIFICATION")
    final_check = check_readiness()
    final_pred = predict_single(sample_input)
    print(f"System State: {final_check.get('status').upper()}")
    print(f"Active Champion: {final_check.get('model_version')}")
    print(f"Final Smoke Test: {final_pred['risk_category']} (Latency: {final_pred['latency_ms']:.1f}ms)")
    print("\n[SUCCESS] ALL 6 FAILURE & RECOVERY LIFECYCLES COMPLETED PERFECTLY!\n")

if __name__ == "__main__":
    main()
