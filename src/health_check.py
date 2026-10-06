import os
import time
import json
import yaml
import requests
from typing import Dict, Any
from src.logger import app_logger, log_failure
from src.predict import load_champion_model, predict_single

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

def load_config(config_path: str = CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def check_liveness() -> Dict[str, Any]:
    """Basic service health check (liveness probe)."""
    return {
        "status": "healthy",
        "service": "credit-risk-api",
        "timestamp": time.time()
    }

def check_readiness(config: dict = None) -> Dict[str, Any]:
    """
    Readiness probe:
    - Verifies champion model and preprocessor artifacts exist and are loadable.
    - Runs a synthetic inference sanity check.
    - Checks latency against max threshold.
    """
    if config is None:
        config = load_config()

    try:
        model, preprocessor, metadata = load_champion_model(force_reload=False, config=config)
        
        # Test synthetic inference
        test_payload = {
            "limit_bal": 20000,
            "sex": 2,
            "education": 2,
            "marriage": 1,
            "age": 25,
            "pay_0": 0,
            "pay_2": 0,
            "pay_3": 0,
            "pay_4": 0,
            "pay_5": 0,
            "pay_6": 0,
            "bill_amt1": 3913,
            "bill_amt2": 3102,
            "bill_amt3": 2680,
            "bill_amt4": 1540,
            "bill_amt5": 1000,
            "bill_amt6": 500,
            "pay_amt1": 689,
            "pay_amt2": 1000,
            "pay_amt3": 500,
            "pay_amt4": 1000,
            "pay_amt5": 1000,
            "pay_amt6": 1000
        }
        pred_res = predict_single(test_payload, config=config)
        latency = pred_res["latency_ms"]
        max_lat = config["monitoring"].get("max_latency_ms", 1000)

        if latency > max_lat:
            app_logger.warning(f"Readiness warning: Test inference latency {latency}ms exceeds threshold {max_lat}ms.")

        return {
            "status": "ready",
            "model_name": metadata.get("model_name", "ChampionModel"),
            "model_version": metadata.get("model_version", "v1.0"),
            "test_latency_ms": latency,
            "quality_gate_passed": metadata.get("quality_gate_passed", True)
        }
    except Exception as e:
        log_failure(
            failure_type="READINESS_PROBE_FAILED",
            severity="CRITICAL",
            component="MODEL_SERVING_LAYER",
            details=f"Model readiness check failed: {str(e)}",
            recommended_action="Verify model artifacts in models/champion/ or trigger automated rollback."
        )
        return {
            "status": "not_ready",
            "error": str(e)
        }

def run_smoke_test(base_url: str = "http://127.0.0.1:8000", timeout: float = 5.0) -> Dict[str, Any]:
    """
    Executes automated post-deployment smoke tests against running API:
    1. GET /health
    2. GET /ready
    3. GET /model-info
    4. POST /predict
    """
    results = {"all_passed": False, "checks": []}
    
    # 1. Health Check
    try:
        r_health = requests.get(f"{base_url}/health", timeout=timeout)
        health_passed = r_health.status_code == 200 and r_health.json().get("status") == "healthy"
        results["checks"].append({"endpoint": "/health", "status_code": r_health.status_code, "passed": health_passed})
    except Exception as e:
        results["checks"].append({"endpoint": "/health", "error": str(e), "passed": False})

    # 2. Ready Check
    try:
        r_ready = requests.get(f"{base_url}/ready", timeout=timeout)
        ready_passed = r_ready.status_code == 200 and r_ready.json().get("status") == "ready"
        results["checks"].append({"endpoint": "/ready", "status_code": r_ready.status_code, "passed": ready_passed})
    except Exception as e:
        results["checks"].append({"endpoint": "/ready", "error": str(e), "passed": False})

    # 3. Model Info Check
    try:
        r_info = requests.get(f"{base_url}/model-info", timeout=timeout)
        info_passed = r_info.status_code == 200 and "model_version" in r_info.json()
        results["checks"].append({"endpoint": "/model-info", "status_code": r_info.status_code, "passed": info_passed})
    except Exception as e:
        results["checks"].append({"endpoint": "/model-info", "error": str(e), "passed": False})

    # 4. Predict Check
    sample_payload = {
        "limit_bal": 50000,
        "sex": 1,
        "education": 1,
        "marriage": 2,
        "age": 28,
        "pay_0": 0,
        "pay_2": 0,
        "pay_3": 0,
        "pay_4": 0,
        "pay_5": 0,
        "pay_6": 0,
        "bill_amt1": 2000,
        "bill_amt2": 1500,
        "bill_amt3": 1200,
        "bill_amt4": 1000,
        "bill_amt5": 800,
        "bill_amt6": 500,
        "pay_amt1": 500,
        "pay_amt2": 500,
        "pay_amt3": 500,
        "pay_amt4": 500,
        "pay_amt5": 500,
        "pay_amt6": 500
    }
    try:
        r_pred = requests.post(f"{base_url}/predict", json=sample_payload, timeout=timeout)
        pred_data = r_pred.json()
        pred_passed = r_pred.status_code == 200 and "prediction" in pred_data and "default_probability" in pred_data
        results["checks"].append({
            "endpoint": "/predict",
            "status_code": r_pred.status_code,
            "passed": pred_passed,
            "latency_ms": pred_data.get("latency_ms", 0)
        })
    except Exception as e:
        results["checks"].append({"endpoint": "/predict", "error": str(e), "passed": False})

    results["all_passed"] = all(c.get("passed", False) for c in results["checks"])
    app_logger.info(f"Smoke test execution summary: {'PASSED' if results['all_passed'] else 'FAILED'}")
    return results

if __name__ == "__main__":
    print("Local readiness check:", check_readiness())
