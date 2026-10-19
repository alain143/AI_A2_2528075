"""Load saved artifacts and produce one prediction. Shared by predict.py and app.py."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .config import FEATURES, MODELS_DIR, RANGES, SYNTHETIC_NOTICE
from .validation import validate_record


class ModelNotFoundError(RuntimeError):
    """Raised when the saved models are missing (run run_all.py first)."""


class Predictor:
    """Loads metadata + three pipelines once; ``predict`` validates then predicts."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        models_dir = Path(models_dir)
        needed = ["metadata.json", "regressor.joblib", "classifier.joblib", "clusterer.joblib"]
        missing = [n for n in needed if not (models_dir / n).exists()]
        if missing:
            raise ModelNotFoundError(f"Missing model files {missing} in {models_dir}. Run: python run_all.py --student-id <id> --seed <LLLL>")
        self.metadata: dict[str, Any] = json.loads((models_dir / "metadata.json").read_text(encoding="utf-8"))
        self.regressor = joblib.load(models_dir / "regressor.joblib")
        self.classifier = joblib.load(models_dir / "classifier.joblib")
        self.clusterer = joblib.load(models_dir / "clusterer.joblib")
        self.verification_code: str = self.metadata["verification_code"]
        self._profiles = {p["cluster"]: p["profile"] for p in self.metadata["cluster_profiles"]}

    def predict(self, record: dict[str, Any]) -> dict[str, Any]:
        clean = validate_record(record)  # raises RecordValidationError
        X = pd.DataFrame([clean], columns=FEATURES)
        yield_tons = float(self.regressor.predict(X)[0])
        prob = float(self.classifier.predict_proba(X)[0, 1])
        flag = int(prob >= 0.5)
        cluster = int(self.clusterer.predict(X)[0])
        return {
            "verification_code": self.verification_code,
            "predicted_yield_tons": round(yield_tons, 4),
            "risk_prediction": "high_risk" if flag else "not_high_risk",
            "high_risk_flag": flag,
            "high_risk_probability": round(prob, 4),
            "cluster": cluster,
            "cluster_profile": self._profiles.get(cluster, "n/a"),
            "human_review_required": True,
            "notice": SYNTHETIC_NOTICE,
        }

    def info(self) -> dict[str, Any]:
        md = self.metadata
        return {
            "verification_code": self.verification_code,
            "student_name": md["student_name"],
            "registration_number": md["registration_number"],
            "seed": md["seed"],
            "dataset_fingerprint_sha256": md["dataset_sha256"],
            "dataset_rows_clean": md["rows_clean"],
            "data_is_synthetic": True,
            "selected_models": md["selected_models"],
            "test_metrics": md["test_metrics"],
            "n_clusters": md["n_clusters"],
            "input_features": FEATURES,
            "input_ranges": {k: list(v) for k, v in RANGES.items()} | {"season": ["A", "B"], "cooperative_support": [0, 1]},
            "library_versions": md["library_versions"],
            "notice": SYNTHETIC_NOTICE,
        }
