import os
import sys
import json
import argparse

# Force UTF-8 on Windows stdout if possible
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.logger import app_logger
from src.data_preprocessing import load_config
from src.train import train_and_compare_models
from src.rollback import deploy_challenger
from src.health_check import check_readiness

def main():
    parser = argparse.ArgumentParser(description="Reliable Credit Risk ML Training Pipeline")
    parser.add_argument("--auto-deploy", action="store_true", default=True, help="Automatically deploy approved challenger to champion")
    args = parser.parse_args()

    print("=" * 70)
    print("[START] STARTING RELIABLE MLOPS RETRAINING & MODEL EVALUATION PIPELINE")
    print("=" * 70)

    config = load_config()

    # Step 1: Ingest, Validate, Preprocess & Train 3 Models
    all_metrics, best_name, metadata = train_and_compare_models(config)

    print("\n" + "=" * 70)
    print("[METRICS] MODEL COMPARISON RESULTS:")
    print("=" * 70)
    for model_name, m in all_metrics.items():
        print(f"* {model_name:22} | Acc: {m['accuracy']:.4f} | Prec: {m['precision']:.4f} | Rec: {m['recall']:.4f} | F1: {m['f1']:.4f} | ROC-AUC: {m['roc_auc']:.4f}")

    print("\n" + "=" * 70)
    print(f"[BEST MODEL SELECTED] {best_name}")
    print(f"Quality Gate Status: {'PASSED' if metadata.get('quality_gate_passed') else 'REJECTED'}")
    print("=" * 70)

    # Step 2: Quality Gate & Champion Deployment
    if metadata.get("quality_gate_passed"):
        if args.auto_deploy:
            print("\nPromoting Challenger to ACTIVE CHAMPION...")
            deploy_challenger(config)
            
            # Step 3: Run Post-deployment Readiness & Sanity Check
            readiness = check_readiness(config)
            print(f"Readiness Sanity Check: {readiness.get('status').upper()} (Latency: {readiness.get('test_latency_ms', 0):.2f}ms)")
            print("\n[SUCCESS] RETRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    else:
        print("\n[WARNING] CHALLENGER FAILED QUALITY GATE. PREVIOUS CHAMPION PRESERVED IN PRODUCTION.")

if __name__ == "__main__":
    main()
