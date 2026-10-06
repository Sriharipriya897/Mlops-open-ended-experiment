import os
import pytest
import numpy as np
from src.evaluate import evaluate_model_performance, explain_recall_significance
from src.train import train_and_compare_models
from src.predict import load_champion_model, predict_single
from src.failure_simulator import simulate_bad_model_training

def test_metrics_calculation():
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 0])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.1, 0.8, 0.7, 0.2])

    metrics = evaluate_model_performance(y_true, y_pred, y_prob)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert metrics["recall"] > 0.50

def test_recall_explanation():
    explanation = explain_recall_significance()
    assert "Recall" in explanation
    assert "Credit Risk" in explanation

def test_model_training_and_comparison():
    all_metrics, best_name, meta = train_and_compare_models()
    assert len(all_metrics) == 3
    assert "Logistic_Regression" in all_metrics
    assert "Random_Forest" in all_metrics
    assert "XGBoost" in all_metrics
    assert best_name in all_metrics
    assert meta["metrics"]["f1"] > 0.40

def test_bad_model_rejection():
    res = simulate_bad_model_training()
    assert res["quality_gate_passed"] is False
    assert res["action_taken"] == "REJECTED_CHAMPION_PRESERVED"
