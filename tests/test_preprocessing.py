import os
import pytest
import numpy as np
import pandas as pd
from src.data_preprocessing import (
    clean_column_names,
    generate_fallback_uci_data,
    validate_data,
    build_preprocessing_pipeline,
    prepare_data,
    load_config
)

def test_generate_data_and_schema():
    df = generate_fallback_uci_data(n_samples=200, random_seed=42)
    assert len(df) == 200
    assert "default" in df.columns
    assert "limit_bal" in df.columns
    assert set(df["default"].unique()).issubset({0, 1})

def test_clean_column_names():
    raw_df = pd.DataFrame({
        "LIMIT_BAL": [20000],
        "default.payment.next.month": [0],
        "PAY_0": [1],
        "ID": [101]
    })
    clean = clean_column_names(raw_df)
    assert "limit_bal" in clean.columns
    assert "default" in clean.columns
    assert "pay_0" in clean.columns
    assert "id" not in clean.columns

def test_validate_data_valid():
    df = generate_fallback_uci_data(n_samples=600, random_seed=42)
    res = validate_data(df)
    assert res["is_valid"] is True
    assert len(res["errors"]) == 0

def test_validate_data_missing_column():
    df = generate_fallback_uci_data(n_samples=600, random_seed=42)
    df_missing = df.drop(columns=["limit_bal"])
    res = validate_data(df_missing)
    assert res["is_valid"] is False
    assert any("limit_bal" in err for err in res["errors"])

def test_validate_data_invalid_target():
    df = generate_fallback_uci_data(n_samples=600, random_seed=42)
    df.loc[0, "default"] = 99 # Invalid class
    res = validate_data(df)
    assert res["is_valid"] is False
    assert any("invalid classes" in err.lower() for err in res["errors"])

def test_preprocessing_pipeline_fit_transform():
    df = generate_fallback_uci_data(n_samples=100, random_seed=42)
    X = df.drop(columns=["default"])
    feature_names = list(X.columns)
    pipeline = build_preprocessing_pipeline(feature_names)
    X_trans = pipeline.fit_transform(X)
    assert X_trans.shape[0] == 100
    assert X_trans.shape[1] > 0
    assert not np.isnan(X_trans).any()
