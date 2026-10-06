import os
import time
import json
import uuid
import yaml
from datetime import datetime
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from src.logger import app_logger, log_prediction, log_failure
from src.predict import predict_single
from src.health_check import check_liveness, check_readiness
from src.rollback import rollback_to_backup, list_backups, deploy_challenger
from src.failure_simulator import update_config_failure_flags, reset_all_failures, simulate_data_drift, simulate_bad_model_training
from src.train import train_and_compare_models
from api.schemas import (
    CreditRiskInput,
    CreditRiskOutput,
    HealthResponse,
    ReadyResponse,
    ModelInfoResponse,
    FailureSimulationRequest,
    RollbackRequest
)
from api.model_loader import model_manager

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Initialize FastAPI App
app = FastAPI(
    title="Reliable Credit Risk Prediction System",
    description="Production-grade MLOps API with automated failure detection, rollback, and recovery.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus Metrics Instrumentation
REQUEST_COUNT = Counter("prediction_requests_total", "Total inference requests", ["status"])
ERROR_COUNT = Counter("prediction_errors_total", "Total inference errors", ["error_type"])
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Inference latency in seconds", buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0])
PREDICTION_DISTRIBUTION = Counter("model_predictions_total", "Distribution of model predictions", ["risk_category"])
MODEL_LOAD_FAILURES = Counter("model_load_failures_total", "Model loading failures")
DATA_DRIFT_GAUGE = Gauge("data_drift_detected", "Data drift alert flag (1 = Drift, 0 = Normal)")

@app.on_event("startup")
async def startup_event():
    """Initializes model loading on API startup."""
    try:
        model_manager.load_model(force_reload=True, enable_auto_failover=True)
        app_logger.info("API Startup: Champion model initialized.")
    except Exception as e:
        MODEL_LOAD_FAILURES.inc()
        app_logger.error(f"API Startup: Model initialization failed ({e}). Will attempt on-demand failover.")

@app.get("/health", response_model=HealthResponse, tags=["Monitoring & Probes"])
async def health_check():
    """Liveness probe to check if the API server process is responding."""
    return check_liveness()

@app.get("/ready", response_model=ReadyResponse, tags=["Monitoring & Probes"])
async def readiness_check():
    """Readiness probe: validates model presence, loading, and inference sanity."""
    res = check_readiness()
    if res.get("status") != "ready":
        return JSONResponse(status_code=503, content=res)
    return res

@app.get("/metrics", tags=["Monitoring & Probes"])
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model Management"])
async def get_model_info():
    """Returns active Champion model metadata, version, parameters, and metrics."""
    try:
        _, _, meta = model_manager.load_model(force_reload=False)
        return {
            "model_name": meta.get("model_name", "ChampionModel"),
            "model_version": meta.get("model_version", "v1.0-default"),
            "created_at": meta.get("created_at"),
            "deployed_at": meta.get("deployed_at"),
            "status": meta.get("status", "ACTIVE_CHAMPION"),
            "metrics": meta.get("metrics", {}),
            "feature_names": meta.get("feature_names", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch model info: {str(e)}")

@app.post("/predict", response_model=CreditRiskOutput, tags=["Inference"])
async def predict_risk(payload: CreditRiskInput):
    """
    Real-time credit risk inference endpoint.
    Predicts default probability and risk category with automated failure handling.
    """
    cfg = load_config()
    start_time = time.perf_counter()

    # Latency metric timer
    with PREDICTION_LATENCY.time():
        try:
            result = predict_single(payload.model_dump(), config=cfg)
            REQUEST_COUNT.labels(status="200").inc()
            PREDICTION_DISTRIBUTION.labels(risk_category=result["risk_category"]).inc()
            return result
        except RuntimeError as re:
            # Simulated API failure or catastrophic error
            ERROR_COUNT.labels(error_type="runtime_error").inc()
            REQUEST_COUNT.labels(status="500").inc()
            raise HTTPException(status_code=500, detail=str(re))
        except ValueError as ve:
            ERROR_COUNT.labels(error_type="validation_error").inc()
            REQUEST_COUNT.labels(status="422").inc()
            raise HTTPException(status_code=422, detail=str(ve))
        except Exception as e:
            ERROR_COUNT.labels(error_type="unhandled_exception").inc()
            REQUEST_COUNT.labels(status="500").inc()
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.get("/drift-status", tags=["Monitoring & Probes"])
async def get_drift_status():
    """Fetches the latest data drift metrics and alert status."""
    cfg = load_config()
    drift_json = os.path.join(BASE_DIR, cfg["paths"]["drift_metrics_json"])
    if os.path.exists(drift_json):
        try:
            with open(drift_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            DATA_DRIFT_GAUGE.set(1 if data.get("dataset_drift_detected") else 0)
            return data
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading drift report: {e}")
    return {"status": "no_drift_report_yet", "dataset_drift_detected": False}

@app.post("/simulate-failure", tags=["Failure & Recovery Simulation"])
async def configure_simulation(flags: FailureSimulationRequest):
    """Enables or disables failure modes (API failure, latency injection, drift)."""
    cfg = update_config_failure_flags(flags.model_dump())
    return {
        "message": "Failure simulation configuration updated.",
        "current_simulation_flags": cfg.get("failure_simulation", {})
    }

@app.post("/reset-failures", tags=["Failure & Recovery Simulation"])
async def reset_failures():
    """Resets all failure simulation flags to normal operating mode."""
    cfg = reset_all_failures()
    return {"message": "All failure simulations cleared. System in normal mode.", "flags": cfg.get("failure_simulation")}

@app.post("/simulate-drift", tags=["Failure & Recovery Simulation"])
async def trigger_drift_simulation():
    """Simulates production distribution shift and updates drift metrics."""
    res = simulate_data_drift()
    DATA_DRIFT_GAUGE.set(1 if res.get("dataset_drift_detected") else 0)
    return {"message": "Drift simulation completed.", "drift_summary": res}

@app.post("/simulate-bad-model", tags=["Failure & Recovery Simulation"])
async def trigger_bad_model():
    """Simulates training a bad challenger and demonstrates rejection by quality gate."""
    res = simulate_bad_model_training()
    return {"message": "Bad model simulation executed.", "result": res}

@app.post("/rollback", tags=["Failure & Recovery Simulation"])
async def trigger_rollback(req: Optional[RollbackRequest] = None):
    """Executes automated or manual rollback to the latest (or selected) model backup."""
    backup_id = req.specific_backup_id if req else None
    res = rollback_to_backup(specific_backup_id=backup_id)
    if res.get("status") == "success":
        model_manager.load_model(force_reload=True)
        return {"message": "Rollback completed successfully.", "details": res}
    else:
        raise HTTPException(status_code=400, detail=res.get("reason", "Rollback failed."))

@app.post("/retrain", tags=["MLOps Pipeline"])
async def trigger_retraining():
    """Executes the end-to-end retraining pipeline and promotes challenger if quality gate passes."""
    try:
        metrics, best_name, meta = train_and_compare_models()
        if meta.get("quality_gate_passed"):
            deploy_challenger()
            model_manager.load_model(force_reload=True)
            return {
                "status": "success",
                "message": f"Retrained and deployed new champion: {best_name}",
                "model_version": meta.get("model_version"),
                "metrics": meta.get("metrics")
            }
        else:
            return {
                "status": "rejected",
                "message": f"Retraining finished, but {best_name} failed quality gate. Champion preserved.",
                "metrics": meta.get("metrics")
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

@app.get("/", response_class=HTMLResponse, tags=["Dashboard UI"])
async def serve_dashboard():
    """Serves the interactive web UI dashboard."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reliable Credit Risk MLOps System</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0f1d;
            --bg-secondary: #111827;
            --bg-card: rgba(17, 24, 39, 0.75);
            --border-color: rgba(255, 255, 255, 0.08);
            --accent-blue: #38bdf8;
            --accent-purple: #a855f7;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-amber: #f59e0b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Outfit', sans-serif;
            background: var(--bg-primary);
            color: var(--text-main);
            min-height: 100vh;
            background-image: radial-gradient(circle at 10% 20%, rgba(56, 189, 248, 0.07) 0%, transparent 40%),
                              radial-gradient(circle at 90% 80%, rgba(168, 85, 247, 0.07) 0%, transparent 40%);
            padding: 24px 32px;
        }

        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 24px;
        }

        .header-title h1 {
            font-size: 24px;
            font-weight: 700;
            background: linear-gradient(135deg, #38bdf8, #a855f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .header-title p {
            font-size: 13px;
            color: var(--text-muted);
            margin-top: 4px;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: #34d399;
            padding: 6px 14px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 600;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: currentColor;
            box-shadow: 0 0 10px currentColor;
        }

        .grid-layout {
            display: grid;
            grid-template-columns: 1.2fr 1fr;
            gap: 24px;
        }

        .card {
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 24px;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 18px;
        }

        .card-title {
            font-size: 16px;
            font-weight: 600;
            color: var(--text-main);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .telemetry-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-bottom: 20px;
        }

        .telemetry-item {
            background: rgba(0, 0, 0, 0.25);
            border: 1px solid var(--border-color);
            padding: 12px 14px;
            border-radius: 10px;
        }

        .telemetry-label {
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .telemetry-val {
            font-size: 17px;
            font-weight: 700;
            margin-top: 4px;
            color: var(--accent-blue);
            font-family: 'JetBrains Mono', monospace;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 14px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        label {
            font-size: 12px;
            color: var(--text-muted);
            font-weight: 500;
        }

        input, select {
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 9px 12px;
            border-radius: 8px;
            font-size: 13px;
            outline: none;
            transition: border-color 0.2s;
        }

        input:focus, select:focus {
            border-color: var(--accent-blue);
            box-shadow: 0 0 8px rgba(56, 189, 248, 0.2);
        }

        .btn {
            background: linear-gradient(135deg, #38bdf8, #2563eb);
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            width: 100%;
            margin-top: 18px;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 8px;
        }

        .btn:hover {
            opacity: 0.95;
            transform: translateY(-1px);
            box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4);
        }

        .sim-btn-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 14px;
        }

        .btn-sim {
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: #f87171;
            padding: 10px 14px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }

        .btn-sim:hover {
            background: rgba(239, 68, 68, 0.25);
            transform: translateY(-1px);
        }

        .btn-recovery {
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: #34d399;
            padding: 10px 14px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }

        .btn-recovery:hover {
            background: rgba(16, 185, 129, 0.25);
            transform: translateY(-1px);
        }

        .result-box {
            background: rgba(0, 0, 0, 0.35);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 16px;
            margin-top: 16px;
            display: none;
        }

        .result-risk {
            font-size: 22px;
            font-weight: 700;
            margin-bottom: 6px;
        }

        .log-console {
            background: #050811;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 12px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            color: #38bdf8;
            height: 140px;
            overflow-y: auto;
            margin-top: 14px;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-title">
            <h1>Reliable Credit Risk Prediction System</h1>
            <p>Automated Failure Detection, Drift Monitoring, Rollback and Recovery Engine</p>
        </div>
        <div class="status-pill" id="system-status-pill">
            <span class="status-dot"></span>
            <span id="system-status-text">SYSTEM HEALTHY</span>
        </div>
    </div>

    <!-- Telemetry Cards -->
    <div class="telemetry-grid">
        <div class="telemetry-item">
            <div class="telemetry-label">Champion Model</div>
            <div class="telemetry-val" id="telemetry-model">v1.0-default</div>
        </div>
        <div class="telemetry-item">
            <div class="telemetry-label">Inference Latency</div>
            <div class="telemetry-val" id="telemetry-latency">-- ms</div>
        </div>
        <div class="telemetry-item">
            <div class="telemetry-label">Data Drift Status</div>
            <div class="telemetry-val" id="telemetry-drift" style="color: var(--accent-green);">NORMAL</div>
        </div>
        <div class="telemetry-item">
            <div class="telemetry-label">API Health</div>
            <div class="telemetry-val" id="telemetry-api" style="color: var(--accent-green);">HEALTHY</div>
        </div>
        <div class="telemetry-item">
            <div class="telemetry-label">Model Quality F1 / AUC</div>
            <div class="telemetry-val" id="telemetry-quality">-- / --</div>
        </div>
        <div class="telemetry-item">
            <div class="telemetry-label">RPO / RTO Target</div>
            <div class="telemetry-val" style="color: #c084fc;">60m / 15m</div>
        </div>
    </div>

    <div class="grid-layout">
        <!-- Left: Prediction Form -->
        <div class="card">
            <div class="card-header">
                <div class="card-title">Customer Risk Assessment Form</div>
                <button type="button" onclick="loadSampleData()" style="background:none; border:1px solid var(--border-color); color:var(--text-muted); padding:4px 8px; border-radius:6px; font-size:12px; cursor:pointer;">Load Default Sample</button>
            </div>
            <form id="prediction-form" onsubmit="submitPrediction(event)">
                <div class="form-grid">
                    <div class="form-group">
                        <label>Credit Limit (NT$)</label>
                        <input type="number" id="limit_bal" value="50000" required>
                    </div>
                    <div class="form-group">
                        <label>Gender</label>
                        <select id="sex">
                            <option value="2">Female</option>
                            <option value="1">Male</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Education</label>
                        <select id="education">
                            <option value="2">University</option>
                            <option value="1">Graduate School</option>
                            <option value="3">High School</option>
                            <option value="4">Other</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Marital Status</label>
                        <select id="marriage">
                            <option value="1">Married</option>
                            <option value="2">Single</option>
                            <option value="3">Other</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Age</label>
                        <input type="number" id="age" value="32" required>
                    </div>
                    <div class="form-group">
                        <label>Payment Status Sept (PAY_0)</label>
                        <select id="pay_0">
                            <option value="0">0 (Paid Duly)</option>
                            <option value="-1">-1 (Pay in Full)</option>
                            <option value="1">1 (Delay 1 mo)</option>
                            <option value="2">2 (Delay 2 mo)</option>
                            <option value="3">3 (Delay 3 mo)</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>PAY_2 (Aug)</label>
                        <select id="pay_2">
                            <option value="0">0</option>
                            <option value="-1">-1</option>
                            <option value="2">2</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Bill Amount Sept</label>
                        <input type="number" id="bill_amt1" value="25000">
                    </div>
                    <div class="form-group">
                        <label>Previous Pay Sept</label>
                        <input type="number" id="pay_amt1" value="2000">
                    </div>
                </div>

                <button type="submit" class="btn" id="btn-submit-pred">
                    <span>Evaluate Credit Risk</span>
                </button>
            </form>

            <div class="result-box" id="result-box">
                <div class="result-risk" id="res-risk-label">--</div>
                <div style="font-size: 13px; color: var(--text-muted);" id="res-details">--</div>
            </div>
        </div>

        <!-- Right: Failure Injection & Recovery Control Center -->
        <div class="card">
            <div class="card-header">
                <div class="card-title">Production Failure Simulation & Self-Healing</div>
                <span style="font-size: 12px; color: #a855f7;">College Demo Mode</span>
            </div>
            <p style="font-size: 13px; color: var(--text-muted);">
                Intentionally test system resilience, automated alert diagnosis, and fast failover recovery.
            </p>

            <div class="sim-btn-grid">
                <button class="btn-sim" onclick="simulateApiFailure()">1. Simulate API Crash</button>
                <button class="btn-sim" onclick="simulateDataDrift()">2. Simulate Data Drift</button>
                <button class="btn-sim" onclick="simulateBadModel()">3. Simulate Bad Model</button>
                <button class="btn-sim" onclick="simulateHighLatency()">4. Simulate High Latency</button>
                <button class="btn-recovery" onclick="triggerRollback()">5. Trigger Rollback</button>
                <button class="btn-recovery" onclick="resetSystem()">6. Reset to Normal</button>
            </div>

            <div class="card-title" style="margin-top: 18px; font-size: 14px;">System Telemetry Log</div>
            <div class="log-console" id="live-console">
                [SYSTEM READY] Listening for telemetry and prediction events...
            </div>
        </div>
    </div>

    <script>
        function logConsole(msg) {
            const el = document.getElementById('live-console');
            const time = new Date().toLocaleTimeString();
            el.innerHTML += `\\n[${time}] ${msg}`;
            el.scrollTop = el.scrollHeight;
        }

        async function updateTelemetry() {
            try {
                const res = await fetch('/model-info');
                if (res.ok) {
                    const data = await res.json();
                    document.getElementById('telemetry-model').innerText = data.model_version || 'v1.0';
                    if (data.metrics && data.metrics.f1) {
                        document.getElementById('telemetry-quality').innerText = `${data.metrics.f1} / ${data.metrics.roc_auc}`;
                    }
                }
                const driftRes = await fetch('/drift-status');
                if (driftRes.ok) {
                    const driftData = await driftRes.json();
                    const elDrift = document.getElementById('telemetry-drift');
                    if (driftData.dataset_drift_detected) {
                        elDrift.innerText = 'DRIFT DETECTED';
                        elDrift.style.color = 'var(--accent-red)';
                    } else {
                        elDrift.innerText = 'NORMAL';
                        elDrift.style.color = 'var(--accent-green)';
                    }
                }
            } catch (e) {
                console.error("Telemetry update error:", e);
            }
        }

        async function submitPrediction(e) {
            e.preventDefault();
            const btn = document.getElementById('btn-submit-pred');
            btn.disabled = true;
            btn.innerText = 'Predicting...';

            const payload = {
                limit_bal: parseFloat(document.getElementById('limit_bal').value),
                sex: parseInt(document.getElementById('sex').value),
                education: parseInt(document.getElementById('education').value),
                marriage: parseInt(document.getElementById('marriage').value),
                age: parseInt(document.getElementById('age').value),
                pay_0: parseInt(document.getElementById('pay_0').value),
                pay_2: parseInt(document.getElementById('pay_2').value),
                pay_3: 0, pay_4: 0, pay_5: 0, pay_6: 0,
                bill_amt1: parseFloat(document.getElementById('bill_amt1').value),
                bill_amt2: 24000, bill_amt3: 23000, bill_amt4: 22000, bill_amt5: 21000, bill_amt6: 20000,
                pay_amt1: parseFloat(document.getElementById('pay_amt1').value),
                pay_amt2: 1500, pay_amt3: 1500, pay_amt4: 1500, pay_amt5: 1500, pay_amt6: 1500
            };

            try {
                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                const box = document.getElementById('result-box');
                box.style.display = 'block';

                if (res.ok) {
                    document.getElementById('telemetry-latency').innerText = `${data.latency_ms} ms`;
                    const riskEl = document.getElementById('res-risk-label');
                    riskEl.innerText = `${data.risk_category} (Default Prob: ${(data.default_probability*100).toFixed(1)}%)`;
                    riskEl.style.color = data.prediction === 1 ? 'var(--accent-red)' : 'var(--accent-green)';
                    document.getElementById('res-details').innerText = `Model: ${data.model_version} | Latency: ${data.latency_ms}ms | ID: ${data.request_id}`;
                    logConsole(`Prediction: ${data.risk_category} (${data.default_probability}) via ${data.model_version} in ${data.latency_ms}ms`);
                } else {
                    document.getElementById('res-risk-label').innerText = 'PREDICTION FAILED';
                    document.getElementById('res-risk-label').style.color = 'var(--accent-red)';
                    document.getElementById('res-details').innerText = data.detail || 'Inference error';
                    logConsole(`ERROR: Prediction failed -> ${data.detail}`);
                }
            } catch (err) {
                logConsole(`NETWORK ERROR: ${err.message}`);
            } finally {
                btn.disabled = false;
                btn.innerText = 'Evaluate Credit Risk';
            }
        }

        async function simulateApiFailure() {
            logConsole("Triggering API Failure Simulation...");
            await fetch('/simulate-failure', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ api_failure: true })
            });
            document.getElementById('telemetry-api').innerText = 'CRASHED';
            document.getElementById('telemetry-api').style.color = 'var(--accent-red)';
            document.getElementById('system-status-text').innerText = 'FAILURE DETECTED';
            document.getElementById('system-status-pill').style.color = 'var(--accent-red)';
            logConsole("ALERT: Model service unavailable! Failover recommended.");
        }

        async function simulateDataDrift() {
            logConsole("Injecting Demographic & Credit Distribution Drift...");
            const res = await fetch('/simulate-drift', { method: 'POST' });
            const data = await res.json();
            logConsole(`DRIFT DETECTED: ${data.drift_summary.drifted_features_count} features drifted. Retraining recommended.`);
            updateTelemetry();
        }

        async function simulateBadModel() {
            logConsole("Simulating Bad Model Training...");
            const res = await fetch('/simulate-bad-model', { method: 'POST' });
            const data = await res.json();
            logConsole(`QUALITY GATE: Crippled candidate rejected (F1: ${data.result.metrics.f1}). Champion preserved.`);
        }

        async function simulateHighLatency() {
            logConsole("Injecting Artificial Latency (+2.0s)...");
            await fetch('/simulate-failure', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ high_latency: true, latency_delay_seconds: 2.0 })
            });
            logConsole("High latency active. Next prediction will experience intentional delay.");
        }

        async function triggerRollback() {
            logConsole("Executing Automated Rollback to last known-good snapshot...");
            const res = await fetch('/rollback', { method: 'POST' });
            const data = await res.json();
            logConsole(`ROLLBACK RESTORED: Restored version ${data.details.model_version}`);
            resetSystem();
        }

        async function resetSystem() {
            await fetch('/reset-failures', { method: 'POST' });
            document.getElementById('telemetry-api').innerText = 'HEALTHY';
            document.getElementById('telemetry-api').style.color = 'var(--accent-green)';
            document.getElementById('system-status-text').innerText = 'SYSTEM HEALTHY';
            document.getElementById('system-status-pill').style.color = '#34d399';
            logConsole("System normalized. All failure flags cleared.");
            updateTelemetry();
        }

        function loadSampleData() {
            document.getElementById('limit_bal').value = 120000;
            document.getElementById('age').value = 45;
            document.getElementById('bill_amt1').value = 85000;
            document.getElementById('pay_amt1').value = 2000;
            document.getElementById('pay_0').value = 2;
            logConsole("Loaded sample high-risk profile data.");
        }

        updateTelemetry();
        setInterval(updateTelemetry, 5000);
    </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
