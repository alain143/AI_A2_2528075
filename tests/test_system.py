"""Tests for the Musanze Harvest Intelligence Service (happy paths AND failure paths)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from tests.helpers import ROOT, SEED, STUDENT_ID, ensure_trained, valid_record

from data.generate_data import base_row_count, generate_dataframe, sha256_of_file, write_csv
from src.config import FEATURES
from src.validation import (
    DataValidationError, IdentityError, RecordValidationError,
    check_leakage, derive_identity, validate_raw_dataset, validate_record, validate_seed,
)


def test_valid_inference_returns_all_fields():
    ensure_trained()
    from src.inference import Predictor

    out = Predictor().predict(valid_record())
    assert out["verification_code"] == "AI-A2-8075"
    assert out["predicted_yield_tons"] > 0
    assert 0.0 <= out["high_risk_probability"] <= 1.0
    assert out["risk_prediction"] in {"high_risk", "not_high_risk"}
    assert isinstance(out["cluster"], int)
    assert out["human_review_required"] is True


def test_missing_field_is_rejected_with_field_name():
    rec = valid_record()
    del rec["field_size_ha"]
    with pytest.raises(RecordValidationError) as err:
        validate_record(rec)
    assert "field_size_ha" in str(err.value)


def test_wrong_type_is_rejected():
    for bad in ("high", True, None):
        rec = valid_record()
        rec["rainfall_mm"] = bad
        with pytest.raises(RecordValidationError):
            validate_record(rec)
    rec = valid_record()
    rec["cooperative_support"] = "yes"
    with pytest.raises(RecordValidationError):
        validate_record(rec)


def test_out_of_range_values_are_rejected():
    rec = valid_record()
    rec["soil_moisture_pct"] = 150
    with pytest.raises(RecordValidationError) as err:
        validate_record(rec)
    assert "soil_moisture_pct" in str(err.value)
    rec = valid_record()
    rec["season"] = "C"
    with pytest.raises(RecordValidationError):
        validate_record(rec)


def test_target_fields_are_rejected_as_leakage():
    rec = valid_record()
    rec["yield_tons"] = 10.0
    with pytest.raises(RecordValidationError):
        validate_record(rec)
    with pytest.raises(DataValidationError):
        check_leakage(FEATURES + ["high_risk"])
    check_leakage(FEATURES)  # the real feature list is clean


def test_dataset_is_reproducible_and_matches_recorded_fingerprint():
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "a.csv", Path(tmp) / "b.csv"
        sha_a = write_csv(generate_dataframe(SEED), a)
        sha_b = write_csv(generate_dataframe(SEED), b)
        assert sha_a == sha_b == sha256_of_file(a)
    df = generate_dataframe(SEED)
    assert len(df) == base_row_count(SEED) + 3 == 282
    recorded = (ROOT / "artifacts" / "data_fingerprint.txt")
    if recorded.exists():
        assert sha_a in recorded.read_text(encoding="utf-8")


def test_raw_dataset_has_the_deliberate_problems_and_validates():
    report = validate_raw_dataset(generate_dataframe(SEED), SEED)
    assert report["missing_by_column"] == {"soil_moisture_pct": 14, "fertilizer_kg": 14}
    assert report["n_duplicate_rows"] == 3
    assert report["matches_expected"] is True
    broken = generate_dataframe(SEED).drop(columns=["season"])
    with pytest.raises(DataValidationError):
        validate_raw_dataset(broken, SEED)


def test_seed_must_equal_last_four_digits():
    assert derive_identity(STUDENT_ID)["verification_code"] == "AI-A2-8075"
    assert validate_seed(STUDENT_ID, 8075)["seed"] == 8075
    with pytest.raises(IdentityError):
        validate_seed(STUDENT_ID, 4671)


def test_saved_metrics_agree_with_test_predictions_csv():
    ensure_trained()
    import numpy as np
    import pandas as pd

    reg = json.loads((ROOT / "artifacts" / "regression_metrics.json").read_text(encoding="utf-8"))
    preds = pd.read_csv(ROOT / "artifacts" / "test_predictions.csv")
    mae = float(np.abs(preds["predicted_yield_tons"] - preds["actual_yield_tons"]).mean())
    assert mae == pytest.approx(reg["models"][reg["selected_model"]]["mae"], abs=1e-3)
    assert set(reg["models"]) >= {"mean_baseline"} and len(reg["models"]) >= 3


def test_api_health_model_info_and_predict():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    ensure_trained()
    from fastapi.testclient import TestClient
    from app import app

    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    info = client.get("/model-info").json()
    assert info["verification_code"] == "AI-A2-8075" and info["data_is_synthetic"] is True
    ok = client.post("/predict", json=valid_record())
    assert ok.status_code == 200 and ok.json()["verification_code"] == "AI-A2-8075"


def test_api_rejects_invalid_requests_with_4xx():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    ensure_trained()
    from fastapi.testclient import TestClient
    from app import app

    client = TestClient(app)
    bad_range = {**valid_record(), "rainfall_mm": 5000}
    bad_type = {**valid_record(), "avg_temp_c": "hot"}
    missing = {k: v for k, v in valid_record().items() if k != "season"}
    extra = {**valid_record(), "yield_tons": 9.9}
    for body in (bad_range, bad_type, missing, extra):
        resp = client.post("/predict", json=body)
        assert resp.status_code == 422
        assert resp.json()["error"] == "validation_failed"
