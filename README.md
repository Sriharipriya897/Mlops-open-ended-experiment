# Reliable Credit Risk Prediction System
### Production-Grade MLOps with Automated Failure Detection, Rollback and Self-Healing Recovery

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)
![MLflow](https://img.shields.io/badge/MLflow-2.10%2B-0194E2.svg)
![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-E6522C.svg)
![Grafana](https://img.shields.io/badge/Grafana-Dashboards-F46800.svg)
![Tests](https://img.shields.io/badge/Pytest-19%20Passed-success.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 1. Problem Statement & Executive Summary

In financial credit risk management, deploying machine learning models to assess whether a loan applicant will default on their payments is mission-critical. A standard ML pipeline frequently experiences severe production failures:
1. **Silent Data & Concept Drift**: Macroeconomic shifts alter customer debt and repayment habits, degrading model accuracy without throwing server exceptions.
2. **Defective Model Deployment**: Inadvertently promoting a challenger model with low Recall leads to catastrophic unrecovered default losses.
3. **API & Infrastructure Crashes**: Model loading corruption or sudden inference crashes disrupt lending operations.
4. **Lack of Automated Rollback**: Manual rollback and incident diagnosis introduce costly downtime.

**The core philosophy of this project is:**
$$\text{Failure} \longrightarrow \text{Detection} \longrightarrow \text{Alert} \longrightarrow \text{Diagnosis} \longrightarrow \text{Failover / Rollback} \longrightarrow \text{Recovery}$$

The system trains and benchmarks 3 ML algorithms (Logistic Regression, Random Forest, and XGBoost), tracks experiments with **MLflow**, enforces pre-deployment **Quality Gates**, serves real-time predictions via **FastAPI**, exports telemetry to **Prometheus & Grafana**, detects statistical distribution drift, and executes **automated zero-downtime rollback** to the last known-good champion model.

---

## 2. System Architecture

```text
                 ┌─────────────────────────────────┐
                 │     GitHub Repository (CI/CD)   │
                 └────────────────┬────────────────┘
                                  │
                                  ▼
                 ┌─────────────────────────────────┐
                 │     Apache Airflow DAG          │
                 │   (Scheduled Retraining Engine) │
                 └────────────────┬────────────────┘
                                  │
                                  ▼
                 ┌─────────────────────────────────┐
                 │ Data Validation & Preprocessing │
                 │ (Scikit-Learn ColumnTransformer)│
                 └────────────────┬────────────────┘
                                  │
                                  ▼
                 ┌─────────────────────────────────┐
                 │  3-Model Candidate Training     │
                 │ (Logistic Regression, RF, XGB)  │
                 └────────────────┬────────────────┘
                                  │
                                  ▼
                 ┌─────────────────────────────────┐
                 │ MLflow Experiment Tracking & Reg│
                 └────────────────┬────────────────┘
                                  │
                                  ▼
                 ┌─────────────────────────────────┐
                 │   Pre-Deployment Quality Gate   │
                 │  (F1, ROC-AUC, Recall Checks)   │
                 └────────────────┬────────────────┘
                                  │
                                  ▼
                 ┌─────────────────────────────────┐
                 │ Champion / Challenger Manager   │
                 │  (Automatic Snapshot Backup)    │
                 └────────────────┬────────────────┘
                                  │
                                  ▼
                 ┌─────────────────────────────────┐
                 │  FastAPI Serving & Web UI       │
                 │ (REST Endpoints & Probes)       │
                 └──────────────┬───┬──────────────┘
                                │   │
                    ┌───────────┘   └───────────┐
                    ▼                           ▼
        ┌───────────────────────┐   ┌────────────────────────┐
        │ Prometheus & Grafana  │   │ Statistical Drift      │
        │ Telemetry Monitoring  │   │ Detector (KS-test/PSI) │
        └───────────┬───────────┘   └───────────┬────────────┘
                    │                           │
                    └───────────┬───────────────┘
                                │ Failure Detected
                                ▼
                 ┌─────────────────────────────────┐
                 │   Alert & Root-Cause Diagnosis  │
                 └────────────────┬────────────────┘
                                │
                                ▼
                 ┌─────────────────────────────────┐
                 │ Automated Rollback & Recovery   │
                 │  (Restore Backup Champion)      │
                 └────────────────┬────────────────┘
                                │
                                ▼
                 ┌─────────────────────────────────┐
                 │ Post-Recovery Readiness Check   │
                 │ (System Health & Smoke Test OK) │
                 └─────────────────────────────────┘
```

---

## 3. Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Language & Core** | Python 3.12, NumPy, Pandas, Scipy |
| **Machine Learning** | Scikit-learn, XGBoost, Joblib |
| **Experiment Tracking** | MLflow |
| **Orchestration** | Apache Airflow |
| **API & Serving** | FastAPI, Uvicorn, Pydantic v2 |
| **Monitoring & Observability** | Prometheus, Grafana, Evidently / Statistical KS-test & PSI |
| **Testing & Quality** | Pytest, Pytest-asyncio, HTTPX |
| **Containerization & CI/CD** | Docker, Docker Compose, GitHub Actions |

---

## 4. Expected Failure & Recovery Matrix

| Failure Scenario | Detection Mechanism | Alert Channel | Automated Diagnosis | Recovery Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **API / Inference Crash** | `/ready` & `/health` probes, HTTP 500 error counts | Prometheus `ModelServiceUnavailable` alert, `logs/failures.log` | `FAILURE_TYPE: SIMULATED_API_FAILURE` | Automated failover restoring last known-good champion backup |
| **Production Data Drift** | Statistical KS-test (p < 0.05) & PSI (> 0.10) | Prometheus `DataDriftDetected` gauge, `reports/drift_report.html` | `FAILURE_TYPE: DATA_DRIFT_DETECTED` | Dispatches alert; triggers Airflow retraining DAG |
| **Bad / Crippled Model** | Pre-deployment Quality Gate (`min_f1`, `min_auc`) | Structured warning, `logs/failures.log` | `FAILURE_TYPE: POOR_MODEL_REJECTED` | Quality Gate rejects challenger; active champion preserved |
| **High Latency Degradation** | Prometheus latency timer (> 1000ms threshold) | Prometheus `HighInferenceLatency` alert | `FAILURE_TYPE: LATENCY_SLA_BREACH` | Clear delay state / thread reset / failover |
| **Data Validation Failure** | Schema & range checks (`validate_data()`) | Airflow pipeline halt, error log | `FAILURE_TYPE: DATA_VALIDATION_FAILURE` | Safe pipeline termination before training starts |

---

## 5. Actual Model Comparison Results

All three models were trained and evaluated on the dataset with stratified validation splits:

| Algorithm | Accuracy | Precision | Recall (Priority #1) | F1-Score | ROC-AUC | Specificity | Training Duration |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.6610** | **0.4176** | **0.5231** | **0.4645** | **0.6540** | **0.7180** | **0.05s** |
| **XGBoost Classifier** | 0.6450 | 0.4000 | 0.5267 | 0.4547 | 0.6320 | 0.6934 | 0.22s |
| **Random Forest** | 0.6880 | 0.4440 | 0.4377 | 0.4409 | 0.6321 | 0.7911 | 0.41s |

> **Why Recall is Critical in Credit Risk**: In lending, a **False Negative** (approving a borrower who subsequently defaults) causes massive, unrecoverable capital losses. Conversely, a **False Positive** (requesting extra documentation from a safe applicant) incurs minor operational friction. High Recall paired with a strong F1-Score protects the lending balance sheet.

---

## 6. Project Structure

```text
reliable-credit-mlops/
├── .github/
│   └── workflows/
│       └── ci.yml                      # GitHub Actions CI/CD pipeline
├── api/
│   ├── main.py                         # FastAPI serving app, Prometheus metrics & UI
│   ├── model_loader.py                 # Thread-safe model manager & failover engine
│   └── schemas.py                      # Pydantic input/output validation schemas
├── config.yaml                         # Central system thresholds & paths
├── dags/
│   └── mlops_retraining_pipeline.py    # Apache Airflow retraining DAG
├── data/
│   ├── backup/                         # Data backups
│   ├── processed/                      # Clean processed data
│   ├── raw/                            # Raw dataset (credit_default.csv)
│   └── reference/                      # Reference distribution for drift checks
├── demo.py                             # Interactive CLI demonstration tool
├── docker-compose.yml                  # Docker Compose multi-service orchestrator
├── Dockerfile                          # Multi-stage container definition
├── docs/
│   ├── architecture.png                # Architecture visual diagram
│   ├── generate_diagrams.py            # Diagram generation utility
│   ├── presentation_content.md         # 15-slide PowerPoint deck content
│   └── project_report.md               # 26-section comprehensive academic report
├── logs/                               # Structured rotating logs (app, predictions, failures)
├── models/
│   ├── backup/                         # Timestamped champion snapshot backups
│   ├── challenger/                     # Newly trained candidate models
│   └── champion/                       # Active production champion model
├── notebooks/
│   └── model_comparison.ipynb         # Jupyter experiment notebook
├── monitoring/
│   ├── alerts.yml                      # Prometheus alert rules
│   ├── prometheus.yml                  # Prometheus scrape config
│   └── grafana/
│       ├── dashboards/                 # 8-panel Grafana dashboard JSON
│       └── provisioning/               # Automated datasource & dashboard configs
├── reports/
│   ├── drift_metrics.json              # Statistical drift JSON
│   ├── drift_report.html               # Interactive HTML drift report
│   ├── metrics.json                    # Calculated model evaluation metrics
│   └── model_comparison.csv            # Tabular model comparison
├── requirements.txt                    # Project dependencies
├── scripts/
│   ├── demo_failures.py                # Automated end-to-end failure & recovery demo
│   ├── enable_failure.py               # CLI tool to toggle failure simulation flags
│   ├── simulate_api_failure.py         # API crash simulation & auto-failover
│   ├── simulate_bad_model.py           # Crippled model quality gate rejection demo
│   ├── simulate_data_drift.py          # Synthetic drift injection & HTML report
│   └── train_pipeline.py               # Standalone ML training & promotion pipeline
└── tests/
    ├── conftest.py                     # Pytest environment setup
    ├── test_api.py                     # API endpoint tests (/health, /ready, /predict)
    ├── test_model.py                   # Training, evaluation & quality gate tests
    ├── test_preprocessing.py           # Preprocessing & schema validation tests
    └── test_rollback.py                # Backup snapshot & rollback recovery tests
```

---

## 7. Windows Setup & Local Execution Guide

### Step 1: Create Virtual Environment (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Step 2: Run Full Model Training Pipeline
Trains all 3 models, verifies quality gates, tracks MLflow runs, and promotes the champion model:
```powershell
python scripts/train_pipeline.py
```

### Step 3: Run Pytest Test Suite
Executes all 19 unit, API, preprocessing, model, and rollback recovery tests:
```powershell
pytest -v tests/
```

### Step 4: Start FastAPI Server & Web Dashboard
```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
* **Interactive Web Dashboard & Demo UI**: Open `http://localhost:8000`
* **Swagger API Documentation**: Open `http://localhost:8000/docs`
* **Prometheus Metrics**: `http://localhost:8000/metrics`
* **Health Check**: `http://localhost:8000/health`
* **Readiness Probe**: `http://localhost:8000/ready`

---

## 8. Interactive Demo & Failure Simulations

### Option A: Interactive Terminal Menu
Run the interactive CLI demonstration:
```powershell
python demo.py
```
```text
=======================================================
   RELIABLE AI - MLOPS FAILURE RECOVERY DEMO
=======================================================
 Current Model : challenger_20261005_071432
 System Status : HEALTHY
=======================================================

 [1] Normal Prediction
 [2] Simulate API Failure (Crash & Failover)
 [3] Simulate Data Drift (KS-test / PSI & Report)
 [4] Simulate Bad Model (Quality Gate Shield)
 [5] Simulate High Latency (SLA Breach)
 [6] Trigger Rollback (Restore Backup Champion)
 [7] Run Health & Readiness Checks
 [8] Run Full End-to-End Recovery Test
 [9] Reset All Failures to Normal
 [0] Exit
=======================================================
```

### Option B: Automated End-to-End Walkthrough
Run all 6 failure and self-healing recovery stages in a single automated script:
```powershell
python scripts/demo_failures.py
```

### Option C: Individual Failure Scenarios
```powershell
# 1. Simulate API Inference Failure & Failover
python scripts/simulate_api_failure.py

# 2. Simulate Production Data Drift & Generate HTML Report
python scripts/simulate_data_drift.py

# 3. Simulate Defective/Corrupted Model Quality Gate Rejection
python scripts/simulate_bad_model.py

# 4. Inject High Latency (+2.0s delay)
python scripts/enable_failure.py --type latency --delay 2.0

# 5. Clear Failure Injections
python scripts/enable_failure.py --type reset
```

---

## 9. Docker & Container Deployment

### Mode 1: Lightweight Mode (Laptop-Friendly)
Runs FastAPI serving and MLflow:
```powershell
docker compose up api mlflow
```

### Mode 2: Full MLOps Stack
Runs FastAPI, MLflow, Prometheus, and Grafana:
```powershell
docker compose up -d
```
* **FastAPI Web UI**: `http://localhost:8000`
* **MLflow Tracking Server**: `http://localhost:5000`
* **Prometheus Telemetry**: `http://localhost:9090`
* **Grafana Dashboards**: `http://localhost:3000` (User: `admin`, Password: `admin`)

---

## 10. Backup Strategy & Disaster Recovery

* **Backup Artifacts**: Complete snapshots of `model.joblib`, `preprocessor.joblib`, and `metadata.json` stored in `models/backup/`.
* **Recovery Point Objective (RPO)**: 60 minutes (frequency of scheduled retraining / dataset sync).
* **Recovery Time Objective (RTO)**: 15 seconds (automated atomic rollback to backup snapshot).

---

## 11. CI/CD Pipeline (GitHub Actions)

Defined in `.github/workflows/ci.yml`:
1. Checks out repository on `push` or `pull_request`.
2. Sets up Python 3.12 environment and installs dependencies.
3. Executes data validation and training pipeline.
4. Runs full 19-test Pytest suite.
5. Executes automated failure and recovery smoke tests.
6. Builds Docker container image.

---

## 12. Limitations & Future Enhancements

* **Current Limitation**: Synthetic data generator is used if external UCI download is network-restricted.
* **Future Enhancement 1**: Automated Canary Deployments (e.g., 90% Champion / 10% Challenger traffic split).
* **Future Enhancement 2**: Real-time SHAP feature importance explainability integrated directly into the web UI.
* **Future Enhancement 3**: Automated Slack / PagerDuty webhook alerting on critical drift detection.

---

## 13. Team Contributions & Academic Statement

* **Dataset Ingestion & Validation Engine**: Data cleaning, schema validation, and Scikit-learn ColumnTransformer.
* **ML Algorithm Benchmarking**: Training Logistic Regression, Random Forest, and XGBoost with MLflow tracking.
* **API Serving & UI**: High-performance FastAPI backend with dark glassmorphic loan evaluation frontend.
* **Reliability & Rollback Framework**: Quality gates, failure simulators, and atomic backup rollback engine.
* **Monitoring & Observability**: Prometheus metrics, Grafana 8-panel dashboard, and statistical drift detection.
