"""Central configuration: paths, feature lists, valid ranges and constants."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"

NUMERIC_FEATURES = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
]
CATEGORICAL_FEATURES = ["season"]
BINARY_FEATURES = ["cooperative_support"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

TARGET_REGRESSION = "yield_tons"
TARGET_CLASSIFICATION = "high_risk"
ALL_COLUMNS = FEATURES + [TARGET_REGRESSION, TARGET_CLASSIFICATION]

# Anything derived from the targets must never be a model input (target leakage).
FORBIDDEN_FEATURES = {"yield_tons", "yield_per_ha", "yield_per_hectare", "high_risk"}

# Inclusive ranges taken from the assignment's generator rules.
RANGES = {
    "rainfall_mm": (600.0, 1400.0),
    "avg_temp_c": (12.0, 26.0),
    "soil_moisture_pct": (20.0, 80.0),
    "fertilizer_kg": (0.0, 250.0),
    "field_size_ha": (0.2, 4.0),
}
SEASONS = ("A", "B")
MIN_YIELD = 0.50

TEST_SIZE = 0.20
CV_FOLDS = 5
K_RANGE = range(2, 7)

SYNTHETIC_NOTICE = (
    "Classroom decision-support prototype trained on SYNTHETIC data. It does not "
    "replace an agronomist and must not be used to decide access to finance, "
    "insurance or public services."
)
