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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.predict import predict_single
from src.health_check import check_readiness, check_liveness
from src.failure_simulator import update_config_failure_flags, reset_all_failures, simulate_data_drift, simulate_bad_model_training
from src.rollback import rollback_to_backup, list_backups
from scripts.demo_failures import main as run_full_recovery_demo

SAMPLE_CUSTOMER = {
    "limit_bal": 50000,
    "sex": 2,
    "education": 2,
    "marriage": 1,
    "age": 32,
    "pay_0": 0,
    "pay_2": 0,
    "pay_3": 0,
    "pay_4": 0,
    "pay_5": 0,
    "pay_6": 0,
    "bill_amt1": 25000,
    "bill_amt2": 24000,
    "bill_amt3": 23000,
    "bill_amt4": 22000,
    "bill_amt5": 21000,
    "bill_amt6": 20000,
    "pay_amt1": 2000,
    "pay_amt2": 1500,
    "pay_amt3": 1500,
    "pay_amt4": 1500,
    "pay_amt5": 1500,
    "pay_amt6": 1500
}

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")

def main():
    while True:
        clear_screen()
        status_info = check_readiness()
        current_status = "HEALTHY" if status_info.get("status") == "ready" else "DEGRADED / CRITICAL"
        current_model = status_info.get("model_version", "unknown")

        print("=" * 55)
        print("   RELIABLE AI - MLOPS FAILURE RECOVERY DEMO")
        print("=" * 55)
        print(f" Current Model : {current_model}")
        print(f" System Status : {current_status}")
        print("=" * 55)
        print("\n [1] Normal Prediction")
        print(" [2] Simulate API Failure")
        print(" [3] Simulate Data Drift")
        print(" [4] Simulate Bad Model (Quality Gate Test)")
        print(" [5] Simulate High Latency")
        print(" [6] Trigger Rollback (Restore Backup Champion)")
        print(" [7] Run Health & Readiness Checks")
        print(" [8] Run Full End-to-End Recovery Test")
        print(" [9] Reset All Failures to Normal")
        print(" [0] Exit")
        print("=" * 55)

        choice = input("\n Select an option [0-9]: ").strip()

        if choice == "1":
            print("\n--- Running Normal Inference ---")
            try:
                res = predict_single(SAMPLE_CUSTOMER)
                print(f"* Request ID   : {res['request_id']}")
                print(f"* Risk Result  : {res['risk_category']} (Prediction: {res['prediction']})")
                print(f"* Default Prob : {res['default_probability']*100:.2f}%")
                print(f"* Model Version: {res['model_version']}")
                print(f"* Latency      : {res['latency_ms']:.2f} ms")
            except Exception as e:
                print(f"[ERROR] Prediction Error: {e}")
            input("\nPress Enter to return to menu...")

        elif choice == "2":
            print("\n--- Simulating API / Inference Failure ---")
            update_config_failure_flags({"api_failure": True})
            print("[ALERT] API Failure Injected! Testing inference now...")
            try:
                predict_single(SAMPLE_CUSTOMER)
            except Exception as e:
                print(f"Expected Failure Caught: {e}")
            print("\nTriggering auto-failover to backup...")
            reset_all_failures()
            rollback_to_backup()
            print("[SUCCESS] System recovered from backup snapshot.")
            input("\nPress Enter to return to menu...")

        elif choice == "3":
            print("\n--- Simulating Production Data Drift ---")
            report = simulate_data_drift()
            print(f"* Drift Detected : {report['dataset_drift_detected']}")
            print(f"* Drift Share    : {report['drift_share']*100:.1f}%")
            print(f"* Drifted Columns: {', '.join(report['drifted_features'][:5])}")
            print("* Report saved   : reports/drift_report.html")
            input("\nPress Enter to return to menu...")

        elif choice == "4":
            print("\n--- Simulating Bad Model Training ---")
            res = simulate_bad_model_training()
            m = res["metrics"]
            print(f"* Candidate F1 : {m['f1']:.4f} (Required: {res['thresholds']['min_f1']})")
            print(f"* Quality Gate : {'PASSED' if res['quality_gate_passed'] else 'REJECTED'}")
            print(f"* Action Taken : {res['action_taken']}")
            input("\nPress Enter to return to menu...")

        elif choice == "5":
            print("\n--- Simulating High Latency (+2.0s delay) ---")
            update_config_failure_flags({"high_latency": True, "latency_delay_seconds": 2.0})
            print("Executing inference with latency injection...")
            res = predict_single(SAMPLE_CUSTOMER)
            print(f"* Observed Latency: {res['latency_ms']:.2f} ms (Exceeds SLA threshold)")
            reset_all_failures()
            print("[SUCCESS] Cleared latency delay.")
            input("\nPress Enter to return to menu...")

        elif choice == "6":
            print("\n--- Triggering Model Rollback ---")
            backups = list_backups()
            print(f"Available backups: {len(backups)}")
            res = rollback_to_backup()
            print(f"Rollback status: {res.get('status')} (Restored: {res.get('model_version')})")
            input("\nPress Enter to return to menu...")

        elif choice == "7":
            print("\n--- Running System Health & Readiness Probes ---")
            live = check_liveness()
            ready = check_readiness()
            print(f"* Liveness  : {live.get('status')}")
            print(f"* Readiness : {ready.get('status')}")
            print(f"* Active Model : {ready.get('model_version')}")
            print(f"* Test Latency : {ready.get('test_latency_ms', 0):.2f} ms")
            input("\nPress Enter to return to menu...")

        elif choice == "8":
            print("\n--- Executing Full End-to-End Recovery Test ---")
            run_full_recovery_demo()
            input("\nPress Enter to return to menu...")

        elif choice == "9":
            reset_all_failures()
            print("\n[SUCCESS] Reset all failures. System running in normal mode.")
            time.sleep(1)

        elif choice == "0":
            print("\nExiting Reliable MLOps CLI. Goodbye!\n")
            break

if __name__ == "__main__":
    main()
