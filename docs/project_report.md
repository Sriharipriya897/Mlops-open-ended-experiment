# Comprehensive Project Report
## Reliable Credit Risk Prediction System with Automated Failure Detection, Rollback and Recovery

---

### 1. Abstract
In financial institutions, deploying Machine Learning (ML) models for credit default risk estimation poses critical operational and financial challenges. A standard ML pipeline frequently fails in production due to silent data drift, infrastructure unreachability, unvalidated challenger model regressions, and abnormal latency. This project presents an end-to-end, resilient MLOps architecture designed around the self-healing principle: **Failure $\rightarrow$ Detection $\rightarrow$ Alert $\rightarrow$ Diagnosis $\rightarrow$ Failover/Rollback $\rightarrow$ Recovery**. The system trains and compares Logistic Regression, Random Forest, and XGBoost models, tracks experiments with MLflow, serves predictions via FastAPI, monitors health with Prometheus and Grafana, tracks distribution shift via statistical tests, and executes automated rollbacks upon quality gate violations or serving failures.

---

### 2. Introduction
Modern machine learning applications require rigorous engineering practices beyond training algorithms in static notebooks. In production, credit scoring APIs must maintain continuous availability and risk fidelity. Unanticipated failures—such as upstream database schema mutations, demographic shifts, or degraded inferences—must be detected in real time and automatically remediated without service interruption.

---

### 3. Problem Statement
Standard credit scoring systems suffer from several points of operational fragility:
1. **Silent Drift**: Changing economic conditions alter customer repayment behavior, eroding model accuracy without raising standard server exceptions.
2. **Unvalidated Deployments**: Inadvertently promoting a challenger model with poor Recall causes massive financial exposure due to undetected defaulters.
3. **Serving Failures**: Memory corruption or missing model weights cause API outages.
4. **Lack of Automated Rollback**: Manual incident recovery introduces hours of downtime and financial losses.

---

### 4. Objectives
* Build an automated credit default prediction system based on the UCI Credit Card Default dataset.
* Train and automatically benchmark three classification algorithms (Logistic Regression, Random Forest, XGBoost) prioritizing Recall, F1-Score, and ROC-AUC.
* Track experiments, hyperparameters, and artifacts using MLflow.
* Enforce strict pre-training data validation guardrails and post-training model quality gates.
* Serve predictions via high-performance FastAPI endpoints with Prometheus metrics.
* Implement statistical drift monitoring (KS-test, PSI) with interactive HTML reports.
* Intentionally simulate and automatically recover from five distinct production failure modes.
* Guarantee automated zero-downtime rollback to the last known-good champion model snapshot.

---

### 5. Dataset Description
* **Dataset Name**: UCI Credit Card Default Dataset
* **Target Feature**: `default` (Binary: `0` = No Default / Safe Borrower, `1` = Default Likely / High Risk)
* **Features (23 attributes)**:
  * Demographic: `LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`
  * Historical Repayment Status (Sept - April): `PAY_0`, `PAY_2`, `PAY_3`, `PAY_4`, `PAY_5`, `PAY_6`
  * Bill Statement Amounts (Sept - April): `BILL_AMT1` to `BILL_AMT6`
  * Previous Payment Amounts (Sept - April): `PAY_AMT1` to `PAY_AMT6`

---

### 6. Data Preprocessing
* **Name Normalization**: Standardizes raw headers into clean lowercase snake_case.
* **Missing Value Imputation**: Median imputation for continuous variables; most frequent imputation for discrete categorical codes.
* **Scaling**: Scikit-learn `StandardScaler` applied to continuous financial variables (`LIMIT_BAL`, `BILL_AMT*`, `PAY_AMT*`, `AGE`).
* **Categorical Encoding**: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` applied to discrete demographics and delay codes.
* **Stratified Splitting**: 80/20 train/test split preserving target class ratio.
* **Artifact Bundling**: Preprocessor ColumnTransformer is serialized with the model artifact (`preprocessor.joblib`).

---

### 7. ML Algorithms
1. **Logistic Regression**: Linear baseline with `class_weight='balanced'` and L2 regularization to balance sensitivity.
2. **Random Forest Classifier**: Non-linear bagging ensemble with 100 estimators, max depth 10, balanced class weighting.
3. **XGBoost Classifier**: Gradient boosted decision trees with `scale_pos_weight=3.0`, learning rate 0.08, logloss objective.

---

### 8. Model Evaluation
The system computes:
* **Accuracy**
* **Precision**
* **Recall (Sensitivity)**: Prioritized because failing to detect an actual defaulter (False Negative) results in direct financial loss.
* **F1-Score**: Harmonic mean of Precision and Recall.
* **ROC-AUC**: Evaluates discrimination capability across risk thresholds.
* **Confusion Matrix**: Saved visually and numerically to `reports/metrics.json` and `reports/model_comparison.csv`.

---

### 9. MLOps Architecture
The system integrates:
1. **GitHub Repository & Actions**: Version control and CI/CD validation.
2. **Airflow DAG**: Automated multi-stage retraining orchestrator.
3. **MLflow**: Centralized run tracking, metric comparison, and artifact storage.
4. **FastAPI Serving Engine**: Containerized REST inference with `/health`, `/ready`, `/predict`, `/metrics`.
5. **Prometheus & Grafana**: Time-series telemetry and real-time dashboard visualization.
6. **Evidently & Statistical Drift Detector**: Continuous distribution comparison against baseline reference data.
7. **Rollback Engine**: Snapshot-based atomic rollback system.

---

### 10. Workflow Automation
Implemented via Apache Airflow (`dags/mlops_retraining_pipeline.py`) with sequential task dependencies:
`validate_raw_data` $\rightarrow$ `preprocess_features` $\rightarrow$ `train_and_evaluate_models` $\rightarrow$ `model_quality_gate` $\rightarrow$ `deploy_and_smoke_test`.

---

### 11. Model Serving
* **Framework**: FastAPI + Uvicorn
* **Latency Guarantee**: < 50ms inference time
* **Input Validation**: Strict Pydantic models preventing invalid inputs.
* **Endpoints**:
  * `GET /`: Interactive web dashboard UI.
  * `GET /health`: Liveness probe.
  * `GET /ready`: Readiness probe verifying model loadability and inference sanity.
  * `GET /metrics`: Prometheus exporter.
  * `POST /predict`: Real-time inference.
  * `GET /model-info`: Current champion metadata.

---

### 12. Monitoring
* **Prometheus Metrics**:
  * `prediction_requests_total`
  * `prediction_errors_total`
  * `prediction_latency_seconds`
  * `model_predictions_total`
  * `data_drift_detected`
  * `model_load_failures_total`
* **Grafana Dashboard**: 8 visual panels displaying real-time traffic, error rates, P95 latencies, drift status, and failure events.

---

### 13. Failure Scenarios
1. **API / Model Crash**: Model loader throws error or endpoint becomes unavailable.
2. **Production Data Drift**: Incoming features deviate from reference distribution by > 20%.
3. **Bad Challenger Model**: Corrupted or underperforming candidate attempts to deploy.
4. **Latency Degradation**: Artificial latency injected exceeding SLA (> 1000ms).
5. **Data Validation Failure**: Malformed schema or missing required columns fed into training pipeline.

---

### 14. Failure Detection
* HTTP 500 / 503 response codes captured by Prometheus counters.
* Readiness probes failing during startup or smoke tests.
* Kolmogorov-Smirnov two-sample test (p-value < 0.05) and PSI (> 0.10).
* Pre-deployment automated quality gate rules.

---

### 15. Alerting
* Prometheus Alertmanager rules configured in `monitoring/alerts.yml`.
* Structured log alerts dispatched to `logs/failures.log` with standardized severity levels.

---

### 16. Diagnosis
Automated diagnostic engine provides actionable, human-readable root-cause summaries:
```text
FAILURE TYPE: DATA_DRIFT_DETECTED
SEVERITY: HIGH
AFFECTED COMPONENT: FEATURE_DISTRIBUTIONS
CURRENT MODEL: challenger_20261005_071432
RECOMMENDED ACTION: Trigger Airflow DAG / Retraining workflow on refreshed distribution.
```

---

### 17. Rollback
* Implemented in `src/rollback.py`.
* Before any challenger deployment, the active champion is snapshotted to `models/backup/backup_<tag>_<timestamp>/`.
* If post-deployment smoke tests or readiness checks fail, `rollback_to_backup()` restores the latest verified snapshot atomically.

---

### 18. Recovery
The recovery lifecycle completes only after:
1. Restoration of the known-good model files.
2. Passing the `/health` and `/ready` probes.
3. Passing a synthetic test prediction within acceptable latency boundaries.

---

### 19. Backup Strategy
* **Snapshots**: Model weights (`model.joblib`), preprocessing transformers (`preprocessor.joblib`), and metadata (`metadata.json`).
* **Recovery Point Objective (RPO)**: 60 minutes.
* **Recovery Time Objective (RTO)**: 15 minutes (automated failover executes in < 2 seconds).

---

### 20. Reliability Principles
1. Never deploy an unvalidated model.
2. Always maintain timestamped snapshots of working champions.
3. Every production model must maintain an immutable version tag.
4. Every failure must produce structured, searchable log records.
5. Automated rollback must be non-destructive.

---

### 21. Testing
* 19 Pytest unit and integration tests across data validation, preprocessing, model training, API endpoints, failure injection, and rollback restoration.
* Automated CI workflow in `.github/workflows/ci.yml`.

---

### 22. Results
* All 3 algorithms successfully trained and compared.
* Best model automatically selected and promoted to Champion.
* 100% test pass rate across the Pytest test suite.
* Automated recovery demonstrated across all failure simulations.

---

### 23. Screenshots & Visual Artifacts
* Architecture Diagram: `docs/architecture.png`
* Data Drift Report: `reports/drift_report.html`
* Metrics Output: `reports/metrics.json`
* Comparison Table: `reports/model_comparison.csv`

---

### 24. Challenges
* Managing Python 3.12 compatibility and platform-specific multiprocessing constraints on Windows.
* Implementing zero-downtime atomic model replacements without restarting the FastAPI server.
* Designing realistic, statistically sound synthetic drift generators.

---

### 25. Future Scope
* Integrating automated canary deployments and shadow testing.
* Connecting Prometheus Alertmanager with Slack/PagerDuty webhooks.
* Adding SHAP (SHapley Additive exPlanations) for real-time model explainability in the UI.

---

### 26. Conclusion
The Reliable Credit Risk Prediction System proves that model accuracy is only one pillar of successful ML engineering. By embedding data validation guardrails, continuous statistical monitoring, quality gates, and automated rollback mechanisms, this project establishes a resilient production-grade MLOps system that survives operational failures autonomously.
