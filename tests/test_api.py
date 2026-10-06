import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

SAMPLE_PAYLOAD = {
    "limit_bal": 50000,
    "sex": 2,
    "education": 2,
    "marriage": 1,
    "age": 30,
    "pay_0": 0,
    "pay_2": 0,
    "pay_3": 0,
    "pay_4": 0,
    "pay_5": 0,
    "pay_6": 0,
    "bill_amt1": 15000,
    "bill_amt2": 14000,
    "bill_amt3": 13000,
    "bill_amt4": 12000,
    "bill_amt5": 11000,
    "bill_amt6": 10000,
    "pay_amt1": 2000,
    "pay_amt2": 2000,
    "pay_amt3": 2000,
    "pay_amt4": 2000,
    "pay_amt5": 2000,
    "pay_amt6": 2000
}

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "credit-risk-api"

def test_ready_endpoint():
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert "model_version" in data

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "prediction_requests_total" in response.text

def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "model_version" in data

def test_predict_endpoint_success():
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in [0, 1]
    assert "default_probability" in data
    assert "risk_category" in data
    assert "latency_ms" in data
    assert data["latency_ms"] < 1000

def test_predict_endpoint_invalid_payload():
    invalid_payload = {"limit_bal": "invalid_string", "age": "abc"}
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
