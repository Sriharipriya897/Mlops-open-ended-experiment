# Presentation Slides: Reliable Credit Risk MLOps System
## Automated Failure Detection, Quality Gates, Rollback and Recovery

---

### Slide 1: Title & Project Overview
* **Title**: Reliable Credit Risk Prediction System with Automated Failure Detection, Rollback and Recovery
* **Subtitle**: A Self-Healing Production MLOps Framework for High-Stakes Financial Lending
* **Presenter**: College MLOps Engineering Demonstration
* **Core Philosophy**: Building ML systems that survive production failures autonomously.

---

### Slide 2: Problem Statement
* Standard ML projects stop at model accuracy in static notebooks.
* Production systems fail due to:
  * Silent data & concept drift
  * Unvalidated model promotions
  * Infrastructure crashes and dependency corruption
  * Latency SLA breaches
* Lack of automated recovery results in extended financial losses and downtime.

---

### Slide 3: Motivation & Business Impact
* In Credit Risk, False Negatives (approving a defaulting borrower) cause direct capital write-offs.
* An undetected 10% drift in default rates can lead to millions in balance sheet exposure.
* Need an autonomous system that detects anomalies, alerts engineers, diagnoses root causes, and executes instant rollbacks.

---

### Slide 4: Project Objectives
* Implement complete ML lifecycle for Credit Risk classification on the UCI Default dataset.
* Benchmark Logistic Regression, Random Forest, and XGBoost models.
* Track experiments and artifacts using MLflow.
* Containerize serving with FastAPI and Prometheus metrics.
* Build automated quality gates and statistical drift detection (KS-test / PSI).
* Intentionally simulate and automatically heal 5 production failure modes.

---

### Slide 5: Dataset & Feature Engineering
* **Dataset**: UCI Credit Card Default Dataset (30,000 credit records).
* **Target**: `default` (Binary: 0 = Safe Borrower, 1 = Default Likely).
* **Feature Dimensions**:
  * Demographics: Credit Limit, Gender, Education, Marriage, Age.
  * Payment History: 6-month repayment delinquency status (`PAY_0` to `PAY_6`).
  * Bill Statement & Payment Amounts: 6-month financial statements and repayments.
* **Pipeline**: Scikit-learn ColumnTransformer with median imputation, one-hot encoding, and standard scaling.

---

### Slide 6: Machine Learning Models & Metric Strategy
* **Algorithms Compared**:
  1. Logistic Regression (Linear baseline + balanced class weighting)
  2. Random Forest Classifier (Non-linear ensemble)
  3. XGBoost Classifier (Gradient boosted trees + positive scale weighting)
* **Metric Prioritization**:
  * **Recall (Sensitivity)**: #1 Priority to catch risky borrowers.
  * **F1-Score**: Balances Precision and Recall.
  * **ROC-AUC**: Evaluates discrimination power.

---

### Slide 7: Model Comparison Results
* **Automated Selection Result**:
  * Logistic Regression: Accuracy: 66.1% | Precision: 41.8% | Recall: 52.3% | F1: 0.465 | ROC-AUC: 0.654
  * Random Forest: Accuracy: 68.8% | Precision: 44.4% | Recall: 43.8% | F1: 0.441 | ROC-AUC: 0.632
  * XGBoost: Accuracy: 64.5% | Precision: 40.0% | Recall: 52.7% | F1: 0.455 | ROC-AUC: 0.632
* Winner promoted based on highest F1-Score & balanced Recall.

---

### Slide 8: End-to-End MLOps Architecture
* **Source Control**: GitHub Repository & Actions CI/CD.
* **Orchestration**: Apache Airflow DAG for scheduled retraining.
* **Tracking & Registry**: MLflow for parameters, metrics, and confusion matrix artifacts.
* **Serving**: FastAPI REST endpoints with Prometheus instrumentation.
* **Observability**: Prometheus scraping + Grafana 8-panel dashboard.
* **Reliability Engine**: Pre-deployment quality gates + atomic snapshot rollback.

---

### Slide 9: Automated Workflow & Quality Gates
* **Data Validation**: Enforces schema presence, non-negative limits, valid target classes.
* **Quality Gate Guardrail**:
  * Requires `F1 >= 0.40`, `ROC-AUC >= 0.60`, `Recall >= 0.40`.
  * If challenger passes $\rightarrow$ Promote to Champion.
  * If challenger fails $\rightarrow$ Reject candidate, preserve existing Champion.

---

### Slide 10: Production Model Serving & Telemetry
* **FastAPI Server**:
  * `GET /health` & `GET /ready` (Liveness & Readiness probes)
  * `GET /metrics` (Prometheus real-time counters and latency histograms)
  * `POST /predict` (Inference with structured logging)
  * `GET /model-info` & `GET /drift-status`
* **Web UI**: Interactive dark-mode dashboard for loan officer evaluation and live failure simulation.

---

### Slide 11: Observability & Monitoring
* **Grafana Dashboard Panels**:
  1. Total API Requests
  2. API Error Rate (%)
  3. Average Latency (ms)
  4. P95 Latency (ms)
  5. Risk Prediction Distribution
  6. Model Version & Health Status
  7. Data Drift Status
  8. Failure Events & Load Errors

---

### Slide 12: Production Failure Scenarios Simulated
1. **API / Model Crash**: Inference engine exception $\rightarrow$ Automated failover to backup model.
2. **Data Drift**: Demographic & debt distribution shifts $\rightarrow$ Statistical KS/PSI alert & HTML report.
3. **Bad Challenger Model**: Corrupted labels $\rightarrow$ Quality gate rejection & champion protection.
4. **Latency Degradation**: SLA breach (> 1000ms) $\rightarrow$ Latency alert.
5. **Data Validation Failure**: Malformed training schema $\rightarrow$ Safe pipeline abortion.

---

### Slide 13: Failure Recovery Lifecycle & Rollback
* **Sequence**:
  * **Failure Occurs** $\rightarrow$ **Detection (Probes/Prometheus)** $\rightarrow$ **Alert Logged** $\rightarrow$ **Diagnosis Generated** $\rightarrow$ **Rollback to Backup Snapshot** $\rightarrow$ **Readiness Smoke Test** $\rightarrow$ **System Recovered**
* **Backup Strategy**: Timestamped snapshots stored in `models/backup/`.
* **RPO**: 60 minutes | **RTO**: < 15 seconds.

---

### Slide 14: Verification, Testing & CI/CD
* **Pytest Suite**: 19 tests covering validation, pipelines, models, APIs, and rollbacks.
* **100% Pass Rate**.
* **GitHub Actions CI/CD**: Automatic build, test, and container packaging on every git push.

---

### Slide 15: Conclusion & Key Takeaways
* Model training is only a fraction of a production ML system.
* Continuous monitoring, automated quality gates, and self-healing rollback mechanisms are mandatory for mission-critical credit risk AI.
* The Reliable Credit Risk MLOps System demonstrates zero-downtime resiliency and robust production engineering.
