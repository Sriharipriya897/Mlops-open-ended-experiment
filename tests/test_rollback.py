import os
import pytest
from src.rollback import backup_champion, rollback_to_backup, list_backups, deploy_challenger
from src.health_check import check_readiness
from src.failure_simulator import update_config_failure_flags, reset_all_failures
from src.predict import predict_single

SAMPLE_PAYLOAD = {
    "limit_bal": 40000,
    "sex": 1,
    "education": 2,
    "marriage": 2,
    "age": 29,
    "pay_0": 0,
    "pay_2": 0,
    "pay_3": 0,
    "pay_4": 0,
    "pay_5": 0,
    "pay_6": 0,
    "bill_amt1": 5000,
    "bill_amt2": 4000,
    "bill_amt3": 3000,
    "bill_amt4": 2000,
    "bill_amt5": 1000,
    "bill_amt6": 500,
    "pay_amt1": 1000,
    "pay_amt2": 1000,
    "pay_amt3": 1000,
    "pay_amt4": 1000,
    "pay_amt5": 1000,
    "pay_amt6": 1000
}

def test_backup_and_list():
    backup_path = backup_champion(tag="pytest_test")
    assert backup_path is not None
    assert os.path.exists(backup_path)

    backups = list_backups()
    assert len(backups) > 0
    assert any("pytest_test" in b["backup_id"] for b in backups)

def test_rollback_restoration():
    rollback_res = rollback_to_backup()
    assert rollback_res["status"] == "success"
    assert "restored_backup" in rollback_res
    
    readiness = check_readiness()
    assert readiness["status"] == "ready"

def test_simulated_api_failure_and_recovery():
    # 1. Enable API failure
    update_config_failure_flags({"api_failure": True})
    
    # 2. Check failure is detected
    readiness_fail = check_readiness()
    assert readiness_fail["status"] == "not_ready"
    
    # 3. Recover
    reset_all_failures()
    rollback_to_backup()
    
    # 4. Verify system restored
    readiness_ok = check_readiness()
    assert readiness_ok["status"] == "ready"
    
    pred_res = predict_single(SAMPLE_PAYLOAD)
    assert pred_res["prediction"] in [0, 1]
