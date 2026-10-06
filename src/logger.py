import os
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

APP_LOG_PATH = os.path.join(LOGS_DIR, "app.log")
PREDICTIONS_LOG_PATH = os.path.join(LOGS_DIR, "predictions.log")
FAILURES_LOG_PATH = os.path.join(LOGS_DIR, "failures.log")

def get_logger(name: str = "reliable_mlops") -> logging.Logger:
    """Returns a configured logger with console and rotating file outputs."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # Rotating file handler for general app logs (max 5MB, up to 5 backups)
    file_handler = RotatingFileHandler(APP_LOG_PATH, maxBytes=5*1024*1024, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger

app_logger = get_logger("mlops_core")

def log_prediction(request_id: str, model_version: str, input_summary: dict, prediction: int, probability: float, latency_ms: float, status_code: int = 200):
    """Logs structured prediction metadata to predictions.log."""
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "request_id": request_id,
        "model_version": model_version,
        "prediction": prediction,
        "probability": round(probability, 4),
        "latency_ms": round(latency_ms, 2),
        "status_code": status_code,
        "features_sample": {k: input_summary[k] for k in list(input_summary.keys())[:5]}
    }
    with open(PREDICTIONS_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    app_logger.info(f"PREDICT request_id={request_id} model={model_version} pred={prediction} prob={probability:.3f} latency={latency_ms:.1f}ms status={status_code}")

def log_failure(failure_type: str, severity: str, component: str, details: str, current_model: str = "unknown", recommended_action: str = "check_logs"):
    """Logs structured failure metadata to failures.log and logs error."""
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "failure_type": failure_type,
        "severity": severity,
        "component": component,
        "current_model": current_model,
        "details": details,
        "recommended_action": recommended_action
    }
    with open(FAILURES_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    
    app_logger.error(
        f"\n{'!'*60}\n"
        f"FAILURE TYPE: {failure_type}\n"
        f"SEVERITY: {severity}\n"
        f"AFFECTED COMPONENT: {component}\n"
        f"CURRENT MODEL: {current_model}\n"
        f"DETAILS: {details}\n"
        f"RECOMMENDED ACTION: {recommended_action}\n"
        f"{'!'*60}"
    )
