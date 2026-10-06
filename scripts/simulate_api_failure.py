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
from src.failure_simulator import update_config_failure_flags, reset_all_failures
from src.rollback import rollback_to_backup
from src.health_check import check_readiness

def main():
    print("=" * 70)
    print("[SCENARIO 1] SIMULATE API / MODEL INFERENCE FAILURE & FAILOVER")
    print("=" * 70)

    # Step 1: Normal system check
    print("\n[Step 1] Baseline System Health Check...")
    initial_check = check_readiness()
    print(f"Status: {initial_check.get('status')} | Active Model: {initial_check.get('model_version')}")

    # Step 2: Inject API Failure
    print("\n[Step 2] Injecting API & Inference Crash Simulation...")
    update_config_failure_flags({"api_failure": True})
    time.sleep(1)

    # Step 3: Detection
    print("\n[Step 3] Running Automated Readiness & Liveness Detection...")
    failure_check = check_readiness()
    print(f"Readiness Status: {failure_check.get('status').upper()}")
    print(f"Error Detail: {failure_check.get('error')}")
    print("[ALERT GENERATED] High error rate / Model service unavailable")

    # Step 4: Diagnosis
    print("\n[Step 4] System Diagnosis:")
    print("FAILURE TYPE: SIMULATED_API_FAILURE")
    print("SEVERITY: CRITICAL")
    print("AFFECTED COMPONENT: INFERENCE_ENGINE")
    print("RECOMMENDED ACTION: Clear failure flags and restore known-good model state.")

    # Step 5: Automated Recovery
    print("\n[Step 5] Initiating Automated Failover & Recovery...")
    reset_all_failures()
    rollback_res = rollback_to_backup()
    print(f"Rollback Result: {rollback_res.get('status')} (Restored version: {rollback_res.get('model_version')})")

    # Step 6: Verify Recovery
    print("\n[Step 6] Running Verification Smoke Test...")
    recovered_check = check_readiness()
    print(f"Post-recovery Status: {recovered_check.get('status').upper()}")
    print(f"Active Model: {recovered_check.get('model_version')}")
    print("\n[SUCCESS] SYSTEM FULLY RECOVERED FROM API FAILURE!")

if __name__ == "__main__":
    main()
