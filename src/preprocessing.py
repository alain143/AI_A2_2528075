"""Cleaning, splitting and the shared preprocessing ColumnTransformer."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET_CLASSIFICATION,
    TARGET_REGRESSION,
    TEST_SIZE,
)


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Read the raw CSV; the index is the zero-based data-row position (row_id)."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}. Run run_all.py first.")
    return pd.read_csv(csv_path)


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop exact duplicate rows (keeping the first). Missing values are NOT filled here:
    imputation is done inside the pipeline so it is fitted on training data only."""
    before = len(df)
    clean = df.drop_duplicates(keep="first").copy()
    clean.index.name = "row_id"
    report = {
        "rows_before": int(before),
        "duplicates_removed": int(before - len(clean)),
        "rows_after": int(len(clean)),
        "missing_after_cleaning": {c: int(clean[c].isna().sum()) for c in FEATURES if clean[c].isna().any()},
        "missing_handling": "median imputation (numeric) fitted on the training split only",
    }
    return clean, report


def make_split(clean: pd.DataFrame, seed: int):
    """Fixed, stratified 80/20 split with random_state = LLLL."""
    train, test = train_test_split(
        clean, test_size=TEST_SIZE, random_state=int(seed), stratify=clean[TARGET_CLASSIFICATION]
    )
    return train, test


def build_preprocessor() -> ColumnTransformer:
    """Impute + scale numerics, impute + one-hot season, impute binary support flag.

    Always used inside a Pipeline so that fitting only ever sees training rows and
    inference applies exactly the same transformations.
    """
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    binary = Pipeline([("impute", SimpleImputer(strategy="most_frequent"))])
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
            ("bin", binary, BINARY_FEATURES),
        ]
    )


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Model inputs only (targets are never included)."""
    return df[FEATURES]
