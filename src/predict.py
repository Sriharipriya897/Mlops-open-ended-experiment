import os
import time
import json
import uuid
import yaml
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, List, Tuple
from datetime import datetime

from src.logger import app_logger, log_prediction, log_failure
from src.data_preprocessing import clean_column_names

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

_MODEL_CACHE = {
    "model": None,
    "preprocessor": None,
    "metadata": None,
    "loaded_time": None
}

def load_config(config_path: str = CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_champion_model(force_reload: bool = False, config: dict = None):
    """Loads active Champion model, preprocessor, and metadata from models/champion/."""
    global _MODEL_CACHE
    if not force_reload and _MODEL_CACHE["model"] is not None:
        return _MODEL_CACHE["model"], _MODEL_CACHE["preprocessor"], _MODEL_CACHE["metadata"]

    if config is None:
        config = load_config()

    champion_dir = os.path.join(BASE_DIR, config["paths"]["champion_model_dir"])
    model_path = os.path.join(champion_dir, "model.joblib")
    prep_path = os.path.join(champion_dir, "preprocessor.joblib")
    meta_path = os.path.join(champion_dir, "metadata.json")

    if not os.path.exists(model_path) or not os.path.exists(prep_path):
        raise FileNotFoundError(f"Champion model or preprocessor not found in {champion_dir}. Run training pipeline first.")

    model = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)
    metadata = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {"model_version": "v1.0-default"}

    _MODEL_CACHE["model"] = model
    _MODEL_CACHE["preprocessor"] = preprocessor
    _MODEL_CACHE["metadata"] = metadata
    _MODEL_CACHE["loaded_time"] = datetime.utcnow()

    app_logger.info(f"Champion model loaded into memory: version={metadata.get('model_version', 'v1.0')}")
    return model, preprocessor, metadata

def predict_single(input_data: Dict[str, Any], config: dict = None) -> Dict[str, Any]:
    """
    Executes real-time credit risk inference for a single customer payload.
    Measures latency and logs structured telemetry.
    """
    start_time = time.perf_counter()
    request_id = str(uuid.uuid4())[:8]

    if config is None:
        config = load_config()

    # Check simulated high latency flag
    sim_cfg = config.get("failure_simulation", {})
    if sim_cfg.get("high_latency", False):
        delay = float(sim_cfg.get("latency_delay_seconds", 2.0))
        time.sleep(delay)

    # Check simulated API/Model failure flag
    if sim_cfg.get("api_failure", False):
        log_failure(
            failure_type="SIMULATED_API_FAILURE",
            severity="CRITICAL",
            component="INFERENCE_ENGINE",
            details="API failure simulation flag is active in config.yaml.",
            recommended_action="Turn off api_failure in failure_simulation or trigger recovery."
        )
        raise RuntimeError("Simulated API failure triggered: Inference engine unavailable.")

    model, preprocessor, metadata = load_champion_model(config=config)
    model_version = metadata.get("model_version", "v1.0")

    # Format input into single-row DataFrame
    df_raw = pd.DataFrame([input_data])
    df_clean = clean_column_names(df_raw)

    # Transform through Scikit-learn Pipeline
    try:
        X_trans = preprocessor.transform(df_clean)
    except Exception as e:
        log_failure(
            failure_type="FEATURE_TRANSFORMATION_ERROR",
            severity="HIGH",
            component="PREPROCESSOR",
            details=f"Failed to transform input features: {str(e)}",
            current_model=model_version,
            recommended_action="Ensure all required schema fields are present in request payload."
        )
        raise ValueError(f"Feature transformation failed: {str(e)}")

    # Model inference
    raw_pred = int(model.predict(X_trans)[0])
    
    prob = 0.5
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X_trans)[0, 1])
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function(X_trans)[0])
        prob = float(1 / (1 + np.exp(-score)))
    else:
        prob = float(raw_pred)

    # Determine risk category
    if prob < 0.30:
        risk_label = "Low Risk"
    elif prob < 0.60:
        risk_label = "Medium Risk"
    else:
        risk_label = "High Risk"

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    # Log prediction event
    log_prediction(
        request_id=request_id,
        model_version=model_version,
        input_summary=input_data,
        prediction=raw_pred,
        probability=prob,
        latency_ms=latency_ms,
        status_code=200
    )

    return {
        "request_id": request_id,
        "prediction": raw_pred,
        "default_probability": round(prob, 4),
        "risk_category": risk_label,
        "model_version": model_version,
        "model_name": metadata.get("model_name", "ChampionModel"),
        "latency_ms": round(latency_ms, 2)
    }

if __name__ == "__main__":
    sample = {
        "limit_bal": 50000,
        "sex": 2,
        "education": 2,
        "marriage": 1,
        "age": 30,
        "pay_0": 2,
        "pay_2": 2,
        "pay_3": 0,
        "pay_4": 0,
        "pay_5": 0,
        "pay_6": 0,
        "bill_amt1": 45000,
        "bill_amt2": 44000,
        "bill_amt3": 43000,
        "bill_amt4": 42000,
        "bill_amt5": 41000,
        "bill_amt6": 40000,
        "pay_amt1": 1000,
        "pay_amt2": 1000,
        "pay_amt3": 1000,
        "pay_amt4": 1000,
        "pay_amt5": 1000,
        "pay_amt6": 1000
    }
    print("Sample inference result:")
    try:
        res = predict_single(sample)
        print(json.dumps(res, indent=2))
    except Exception as e:
        print(f"Prediction error: {e}")
