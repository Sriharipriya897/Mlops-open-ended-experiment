from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class CreditRiskInput(BaseModel):
    limit_bal: float = Field(..., description="Amount of given credit in NT dollars")
    sex: int = Field(..., description="Gender (1=male, 2=female)")
    education: int = Field(..., description="Education (1=grad school, 2=uni, 3=high school, 4=other)")
    marriage: int = Field(..., description="Marital status (1=married, 2=single, 3=others)")
    age: int = Field(..., description="Age in years")
    pay_0: int = Field(..., description="Repayment status in Sept (-1=pay duly, 1=delay 1 mo, 2=delay 2 mo)")
    pay_2: int = Field(..., description="Repayment status in Aug")
    pay_3: int = Field(..., description="Repayment status in July")
    pay_4: int = Field(..., description="Repayment status in June")
    pay_5: int = Field(..., description="Repayment status in May")
    pay_6: int = Field(..., description="Repayment status in April")
    bill_amt1: float = Field(..., description="Bill statement amount in Sept")
    bill_amt2: float = Field(..., description="Bill statement amount in Aug")
    bill_amt3: float = Field(..., description="Bill statement amount in July")
    bill_amt4: float = Field(..., description="Bill statement amount in June")
    bill_amt5: float = Field(..., description="Bill statement amount in May")
    bill_amt6: float = Field(..., description="Bill statement amount in April")
    pay_amt1: float = Field(..., description="Previous payment amount in Sept")
    pay_amt2: float = Field(..., description="Previous payment amount in Aug")
    pay_amt3: float = Field(..., description="Previous payment amount in July")
    pay_amt4: float = Field(..., description="Previous payment amount in June")
    pay_amt5: float = Field(..., description="Previous payment amount in May")
    pay_amt6: float = Field(..., description="Previous payment amount in April")

    model_config = {
        "json_schema_extra": {
            "example": {
                "limit_bal": 50000,
                "sex": 2,
                "education": 2,
                "marriage": 1,
                "age": 35,
                "pay_0": 0,
                "pay_2": 0,
                "pay_3": 0,
                "pay_4": 0,
                "pay_5": 0,
                "pay_6": 0,
                "bill_amt1": 3913,
                "bill_amt2": 3102,
                "bill_amt3": 2680,
                "bill_amt4": 1540,
                "bill_amt5": 1000,
                "bill_amt6": 500,
                "pay_amt1": 689,
                "pay_amt2": 1000,
                "pay_amt3": 500,
                "pay_amt4": 1000,
                "pay_amt5": 1000,
                "pay_amt6": 1000
            }
        }
    }

class CreditRiskOutput(BaseModel):
    request_id: str
    prediction: int = Field(..., description="0 = No Default, 1 = Default Likely")
    default_probability: float
    risk_category: str
    model_version: str
    model_name: str
    latency_ms: float

class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: float

class ReadyResponse(BaseModel):
    status: str
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    test_latency_ms: Optional[float] = None
    quality_gate_passed: Optional[bool] = None
    error: Optional[str] = None

class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    created_at: Optional[str] = None
    deployed_at: Optional[str] = None
    status: str
    metrics: Dict[str, Any]
    feature_names: List[str]

class FailureSimulationRequest(BaseModel):
    api_failure: Optional[bool] = False
    high_latency: Optional[bool] = False
    latency_delay_seconds: Optional[float] = 2.0
    data_drift: Optional[bool] = False
    bad_model: Optional[bool] = False

class RollbackRequest(BaseModel):
    specific_backup_id: Optional[str] = None
