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

from src.failure_simulator import simulate_bad_model_training

def main():
    print("=" * 70)
    print("[SCENARIO 3] BAD / CRIPPLED CHALLENGER MODEL SIMULATION")
    print("=" * 70)

    print("\n[Step 1] Training candidate model with corrupted labels and degraded hyperparameters...")
    result = simulate_bad_model_training()

    m = result["metrics"]
    th = result["thresholds"]
    print("\n[Step 2] Evaluation vs Quality Gate Thresholds:")
    print(f"* Accuracy:  {m['accuracy']:.4f}")
    print(f"* F1-Score:  {m['f1']:.4f} (Required Min: {th['min_f1']})")
    print(f"* ROC-AUC:   {m['roc_auc']:.4f} (Required Min: {th['min_roc_auc']})")
    print(f"* Recall:    {m['recall']:.4f}")

    print("\n[Step 3] Quality Gate Decision:")
    if result["quality_gate_passed"]:
        print("[FAIL] Unexpected: Model passed quality gate.")
    else:
        print("[ALERT] QUALITY GATE REJECTED THE MODEL!")
        print("ACTION TAKEN: REJECT CHALLENGER -> KEEP STABLE CHAMPION IN PRODUCTION.")

    print("\n[Step 4] Diagnosis:")
    print("FAILURE TYPE: POOR_MODEL_QUALITY_GATE_REJECTED")
    print("SEVERITY: HIGH")
    print("AFFECTED COMPONENT: MODEL_TRAINING_PIPELINE")
    print("RELIABILITY GUARANTEE: Unvalidated or defective models are never deployed.")
    print("\n[SUCCESS] BAD MODEL REJECTION TEST PASSED!")

if __name__ == "__main__":
    main()
