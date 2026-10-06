import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
import time
import json
import yaml
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, Optional
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import mlflow
import mlflow.sklearn
import mlflow.xgboost

from src.logger import app_logger, log_failure
from src.data_preprocessing import prepare_data, load_config
from src.evaluate import evaluate_model_performance, save_comparison_reports

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

def setup_mlflow(config: dict) -> str:
    """Initializes MLflow tracking URI and active experiment."""
    mlflow_cfg = config.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "")
    experiment_name = mlflow_cfg.get("experiment_name", "credit-risk-production")

    if not tracking_uri or tracking_uri.startswith("sqlite:///"):
        # Use file-based tracking for maximum local speed and zero SQLite lock issues
        mlruns_dir = os.path.join(BASE_DIR, "mlruns")
        os.makedirs(mlruns_dir, exist_ok=True)
        tracking_uri = f"file:///{mlruns_dir.replace(os.sep, '/')}"

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        app_logger.info(f"MLflow configured: URI={tracking_uri}, Experiment={experiment_name}")
    except Exception as e:
        app_logger.warning(f"MLflow setup warning: {e}")
    return experiment_name

def save_confusion_matrix_plot(cm_dict: dict, model_name: str, output_path: str):
    """Generates and saves a confusion matrix visualization."""
    cm = np.array([
        [cm_dict["true_negatives"], cm_dict["false_positives"]],
        [cm_dict["false_negatives"], cm_dict["true_positives"]]
    ])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Non-Default (0)", "Default (1)"],
                yticklabels=["Non-Default (0)", "Default (1)"])
    plt.title(f"Confusion Matrix: {model_name}")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def train_and_compare_models(config: dict = None) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    """
    Trains 3 ML algorithms:
    1. Logistic Regression
    2. Random Forest
    3. XGBoost
    Evaluates each model, tracks via MLflow, and selects the winning candidate.
    """
    if config is None:
        config = load_config()

    setup_mlflow(config)
    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data(config)

    models = {
        "Logistic_Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=config["project"]["random_seed"]
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            class_weight="balanced",
            random_state=config["project"]["random_seed"],
            n_jobs=1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=3.0,
            eval_metric="logloss",
            random_state=config["project"]["random_seed"]
        )
    }

    all_metrics = {}
    trained_artifacts = {}
    temp_plots_dir = os.path.join(BASE_DIR, "reports", "plots")
    os.makedirs(temp_plots_dir, exist_ok=True)

    app_logger.info("Starting training runs for 3 ML candidate models...")

    for model_name, model_obj in models.items():
        start_time = time.time()
        app_logger.info(f"Training model: {model_name}...")
        
        # Fit model
        model_obj.fit(X_train, y_train)
        train_duration = time.time() - start_time

        # Predict & Evaluate
        y_pred = model_obj.predict(X_test)
        y_prob = None
        if hasattr(model_obj, "predict_proba"):
            y_prob = model_obj.predict_proba(X_test)[:, 1]

        metrics = evaluate_model_performance(y_test, y_pred, y_prob)
        metrics["training_duration_seconds"] = round(train_duration, 3)
        all_metrics[model_name] = metrics
        trained_artifacts[model_name] = model_obj

        # Save confusion matrix plot
        plot_path = os.path.join(temp_plots_dir, f"cm_{model_name.lower()}.png")
        save_confusion_matrix_plot(metrics["confusion_matrix"], model_name, plot_path)

        # Log run in MLflow
        try:
            with mlflow.start_run(run_name=f"run_{model_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"):
                # Log model parameters
                if hasattr(model_obj, "get_params"):
                    mlflow.log_params({k: v for k, v in model_obj.get_params().items() if isinstance(v, (int, float, str, bool))})
                
                # Log metrics
                mlflow.log_metrics({
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1"],
                    "roc_auc": metrics["roc_auc"],
                    "specificity": metrics["specificity"],
                    "train_duration_sec": train_duration
                })

                # Log tags & artifacts
                mlflow.set_tag("model_type", model_name)
                mlflow.set_tag("dataset_version", "v1.0-uci-default")
                if os.path.exists(plot_path):
                    mlflow.log_artifact(plot_path)

                # Log model artifact
                model_artifact_path = os.path.join(temp_plots_dir, f"{model_name.lower()}_model.joblib")
                joblib.dump(model_obj, model_artifact_path)
                mlflow.log_artifact(model_artifact_path)
                    
                app_logger.info(f"MLflow run logged for {model_name}. F1={metrics['f1']}, ROC-AUC={metrics['roc_auc']}, Recall={metrics['recall']}")
        except Exception as e:
            app_logger.warning(f"MLflow run logging skipped or encountered warning: {e}")

    # Save comprehensive comparison reports
    save_comparison_reports(all_metrics, config)

    # Automated Best Model Selection (Prioritizing F1-Score and Recall for Credit Risk)
    best_model_name = max(all_metrics.keys(), key=lambda m: (all_metrics[m]["f1"], all_metrics[m]["roc_auc"], all_metrics[m]["recall"]))
    best_metrics = all_metrics[best_model_name]
    best_model_obj = trained_artifacts[best_model_name]

    app_logger.info(f"Winning candidate selected: {best_model_name} (F1={best_metrics['f1']}, Recall={best_metrics['recall']}, ROC-AUC={best_metrics['roc_auc']})")

    # Quality Gate Check
    quality_cfg = config["model"]
    min_f1 = quality_cfg.get("minimum_f1", 0.45)
    min_auc = quality_cfg.get("minimum_roc_auc", 0.70)
    min_recall = quality_cfg.get("minimum_recall", 0.40)

    quality_passed = (
        best_metrics["f1"] >= min_f1 and
        best_metrics["roc_auc"] >= min_auc and
        best_metrics["recall"] >= min_recall
    )

    challenger_dir = os.path.join(BASE_DIR, config["paths"]["challenger_model_dir"])
    os.makedirs(challenger_dir, exist_ok=True)

    challenger_metadata = {
        "model_name": best_model_name,
        "model_version": f"challenger_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "metrics": best_metrics,
        "feature_names": feature_names,
        "quality_gate_passed": quality_passed,
        "quality_gate_thresholds": {
            "minimum_f1": min_f1,
            "minimum_roc_auc": min_auc,
            "minimum_recall": min_recall
        }
    }

    if quality_passed:
        # Save challenger model and metadata
        challenger_model_path = os.path.join(challenger_dir, "model.joblib")
        challenger_preprocessor_path = os.path.join(challenger_dir, "preprocessor.joblib")
        challenger_meta_path = os.path.join(challenger_dir, "metadata.json")

        joblib.dump(best_model_obj, challenger_model_path)
        joblib.dump(preprocessor, challenger_preprocessor_path)
        with open(challenger_meta_path, "w", encoding="utf-8") as f:
            json.dump(challenger_metadata, f, indent=2)

        app_logger.info(f"QUALITY GATE PASSED. Model approved as Challenger in {challenger_dir}")
    else:
        log_failure(
            failure_type="MODEL_QUALITY_GATE_REJECTED",
            severity="HIGH",
            component="MODEL_TRAINING_PIPELINE",
            details=f"Best model {best_model_name} failed quality gate. F1={best_metrics['f1']} (min {min_f1}), ROC-AUC={best_metrics['roc_auc']} (min {min_auc})",
            current_model="previous_champion",
            recommended_action="Keep current champion active. Review data quality or retrain with adjusted hyperparameters."
        )

    return all_metrics, best_model_name, challenger_metadata

if __name__ == "__main__":
    metrics, best_name, meta = train_and_compare_models()
    print(f"\nFinal Best Model: {best_name}")
    print(json.dumps(meta, indent=2))
