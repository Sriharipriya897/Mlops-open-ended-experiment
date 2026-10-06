import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.failure_simulator import update_config_failure_flags, reset_all_failures

def main():
    parser = argparse.ArgumentParser(description="Configure MLOps Failure Simulation Modes")
    parser.add_argument("--type", choices=["latency", "api", "drift", "bad_model", "reset"], required=True, help="Failure scenario type")
    parser.add_argument("--delay", type=float, default=2.0, help="Latency delay in seconds (for latency type)")
    args = parser.parse_args()

    if args.type == "latency":
        update_config_failure_flags({"high_latency": True, "latency_delay_seconds": args.delay})
        print(f"✅ High latency simulation ENABLED (Delay: {args.delay}s). Next inferences will be artificially delayed.")
    elif args.type == "api":
        update_config_failure_flags({"api_failure": True})
        print("🚨 API failure simulation ENABLED. Next inferences will trigger error responses.")
    elif args.type == "drift":
        update_config_failure_flags({"data_drift": True})
        print("📈 Data drift flag ENABLED.")
    elif args.type == "bad_model":
        update_config_failure_flags({"bad_model": True})
        print("🛑 Bad model flag ENABLED.")
    elif args.type == "reset":
        reset_all_failures()
        print("✅ ALL failure simulations CLEARED. System restored to normal operating state.")

if __name__ == "__main__":
    main()
