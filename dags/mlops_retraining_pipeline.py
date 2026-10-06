"""
Apache Airflow DAG: Automated Credit Risk Retraining & Self-Healing Pipeline
Workflow:
Data Validation -> Preprocessing -> Train Models (LR, RF, XGB) -> Evaluate & Compare ->
Quality Gate -> Register Challenger -> Deploy Challenger -> Post-deployment Smoke Test -> Promote Champion
"""

import os
import sys
from datetime import datetime, timedelta

# Airflow imports with standalone fallback compatibility
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.operators.empty import EmptyOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.logger import app_logger
from src.data_preprocessing import prepare_data, load_config, ensure_raw_data, validate_data
from src.train import train_and_compare_models
from src.rollback import deploy_challenger, rollback_to_backup
from src.health_check import check_readiness

default_args = {
    "owner": "mlops_team",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

def task_validate_data(**kwargs):
    config = load_config()
    df = ensure_raw_data(config)
    report = validate_data(df, config)
    if not report["is_valid"]:
        raise ValueError(f"Data validation failed: {report['errors']}")
    app_logger.info("Airflow Task: Data Validation Passed.")
    return True

def task_preprocess_data(**kwargs):
    config = load_config()
    prepare_data(config)
    app_logger.info("Airflow Task: Data Preprocessing & Splitting Complete.")
    return True

def task_train_and_evaluate(**kwargs):
    config = load_config()
    all_metrics, best_name, metadata = train_and_compare_models(config)
    app_logger.info(f"Airflow Task: Models Trained. Winning Candidate: {best_name}")
    return metadata

def task_quality_gate(**kwargs):
    config = load_config()
    # Check challenger metadata
    challenger_meta_path = os.path.join(BASE_DIR, config["paths"]["challenger_model_dir"], "metadata.json")
    if not os.path.exists(challenger_meta_path):
        raise FileNotFoundError("Challenger metadata missing.")
    
    import json
    with open(challenger_meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    if not meta.get("quality_gate_passed", False):
        raise ValueError(f"Challenger failed Quality Gate. Metrics: {meta.get('metrics')}")
    
    app_logger.info("Airflow Task: Quality Gate Verified.")
    return True

def task_deploy_and_smoke_test(**kwargs):
    config = load_config()
    # Step 1: Deploy challenger to champion (with auto-backup)
    success = deploy_challenger(config)
    if not success:
        raise RuntimeError("Failed to deploy challenger.")

    # Step 2: Post-deployment Readiness & Smoke Test
    readiness = check_readiness(config)
    if readiness.get("status") != "ready":
        app_logger.error("Airflow: Smoke test failed after deployment! Initiating auto-rollback...")
        rollback_to_backup(config)
        raise RuntimeError(f"Deployment smoke test failed. Rolled back to previous champion. Error: {readiness.get('error')}")

    app_logger.info(f"Airflow Task: Model Deployed & Verified. Active: {readiness.get('model_version')}")
    return True

if AIRFLOW_AVAILABLE:
    dag = DAG(
        "credit_risk_mlops_retraining",
        default_args=default_args,
        description="Automated MLOps Retraining, Evaluation, Quality Gate & Safe Deployment Pipeline",
        schedule_interval="@weekly",
        catchup=False,
        tags=["mlops", "credit-risk", "production"],
    )

    start_op = EmptyOperator(task_id="start_pipeline", dag=dag)
    
    validate_op = PythonOperator(
        task_id="validate_raw_data",
        python_callable=task_validate_data,
        dag=dag,
    )

    preprocess_op = PythonOperator(
        task_id="preprocess_features",
        python_callable=task_preprocess_data,
        dag=dag,
    )

    train_eval_op = PythonOperator(
        task_id="train_and_evaluate_models",
        python_callable=task_train_and_evaluate,
        dag=dag,
    )

    quality_gate_op = PythonOperator(
        task_id="model_quality_gate",
        python_callable=task_quality_gate,
        dag=dag,
    )

    deploy_test_op = PythonOperator(
        task_id="deploy_and_smoke_test",
        python_callable=task_deploy_and_smoke_test,
        dag=dag,
    )

    end_op = EmptyOperator(task_id="end_pipeline", dag=dag)

    start_op >> validate_op >> preprocess_op >> train_eval_op >> quality_gate_op >> deploy_test_op >> end_op

if __name__ == "__main__":
    print("Executing Airflow DAG tasks in standalone script mode:")
    task_validate_data()
    task_preprocess_data()
    task_train_and_evaluate()
    task_quality_gate()
    task_deploy_and_smoke_test()
    print("✅ All DAG tasks executed successfully in standalone mode!")
