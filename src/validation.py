"""Identity, dataset and single-record validation with helpful error messages."""
from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd

from .config import (
    ALL_COLUMNS,
    FEATURES,
    FORBIDDEN_FEATURES,
    MIN_YIELD,
    NUMERIC_FEATURES,
    RANGES,
    SEASONS,
)


class ValidationError(ValueError):
    """Base class; ``errors`` holds one human-readable message per problem."""

    def __init__(self, errors: list[str] | str):
        self.errors = [errors] if isinstance(errors, str) else list(errors)
        super().__init__("; ".join(self.errors))


class IdentityError(ValidationError):
    """Student id / seed problem."""


class DataValidationError(ValidationError):
    """Dataset-level problem (schema, range, label values)."""


class RecordValidationError(ValidationError):
    """Single prediction record problem."""


# --------------------------------------------------------------------- identity
def derive_identity(student_id: str) -> dict[str, Any]:
    """Derive REGDIGITS, LLLL, seed and verification code from the institutional username.

    ``ug2528075@ines.ac.rw`` -> regdigits ``2528075``, LLLL ``8075``, code ``AI-A2-8075``.
    """
    if not isinstance(student_id, str) or not student_id.strip():
        raise IdentityError("--student-id must be a non-empty string such as ug2528075@ines.ac.rw")
    username = student_id.strip().split("@")[0]
    digits = "".join(re.findall(r"\d", username))
    if len(digits) < 4:
        raise IdentityError(
            f"--student-id '{student_id}' must contain the registration digits (at least four)."
        )
    last4 = digits[-4:]
    return {
        "student_id": student_id.strip(),
        "regdigits": digits,
        "last4": last4,
        "seed": int(last4),
        "verification_code": f"AI-A2-{last4}",
    }


def validate_seed(student_id: str, seed: int) -> dict[str, Any]:
    """Reject any seed that is not LLLL (last four registration digits)."""
    ident = derive_identity(student_id)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise IdentityError(f"--seed must be an integer, got {seed!r}.")
    if seed != ident["seed"]:
        raise IdentityError(
            f"Seed {seed} rejected: the seed must be LLLL = {ident['seed']} "
            f"(last four digits of {ident['regdigits']})."
        )
    return ident


# ---------------------------------------------------------------------- dataset
def check_leakage(feature_names: list[str]) -> None:
    """Raise if any feature is (derived from) a target."""
    bad = [f for f in feature_names if f in FORBIDDEN_FEATURES or "yield" in f.lower()]
    if bad:
        raise DataValidationError(f"Target leakage: {bad} must not be model inputs.")


def validate_raw_dataset(df: pd.DataFrame, seed: int | None = None) -> dict[str, Any]:
    """Validate the raw CSV and return an evidence report.

    Hard problems (schema, ranges, labels) raise ``DataValidationError``.
    Missing values and duplicates are expected, so they are *reported* and handled
    later by the cleaning/preprocessing steps.
    """
    errors: list[str] = []
    missing_cols = [c for c in ALL_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing required columns: {missing_cols}")

    for col in NUMERIC_FEATURES + ["yield_tons"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Column '{col}' must be numeric.")
    if errors:
        raise DataValidationError(errors)

    for col, (lo, hi) in RANGES.items():
        bad = df[col].dropna()
        bad = bad[(bad < lo - 1e-9) | (bad > hi + 1e-9)]
        if len(bad):
            errors.append(f"{col}: {len(bad)} value(s) outside [{lo}, {hi}].")
    if (df["yield_tons"] < MIN_YIELD - 1e-9).any():
        errors.append(f"yield_tons has values below {MIN_YIELD}.")
    if not df["season"].dropna().isin(SEASONS).all():
        errors.append("season must be 'A' or 'B'.")
    if not df["cooperative_support"].dropna().isin([0, 1]).all():
        errors.append("cooperative_support must be 0 or 1.")
    if not df["high_risk"].dropna().isin([0, 1]).all():
        errors.append("high_risk must be 0 or 1.")
    for col in ["season", "cooperative_support", "yield_tons", "high_risk"]:
        if df[col].isna().any():
            errors.append(f"Unexpected missing values in '{col}'.")
    if errors:
        raise DataValidationError(errors)

    check_leakage(FEATURES)

    report: dict[str, Any] = {
        "n_rows_raw": int(len(df)),
        "missing_by_column": {c: int(df[c].isna().sum()) for c in ALL_COLUMNS if df[c].isna().any()},
        "n_duplicate_rows": int(df.duplicated().sum()),
        "leakage_check": "passed: no yield_tons / yield_per_ha / high_risk in model inputs",
    }
    if seed is not None:
        n_base = 240 + (int(seed) % 41)
        m = int(round(0.05 * n_base))
        report["expected"] = {"base_rows": n_base, "raw_rows": n_base + 3, "missing_per_column": m, "duplicates": 3}
        report["matches_expected"] = bool(
            len(df) == n_base + 3
            and report["n_duplicate_rows"] == 3
            and report["missing_by_column"].get("soil_moisture_pct") == m
            and report["missing_by_column"].get("fertilizer_kg") == m
        )
    return report


# ----------------------------------------------------------------------- record
def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_record(record: Any) -> dict[str, Any]:
    """Validate one prediction record (CLI path; the API also uses pydantic)."""
    if not isinstance(record, dict):
        raise RecordValidationError("Input must be a single JSON object of farm fields.")
    errors: list[str] = []

    for name in FEATURES:
        if name not in record:
            errors.append(f"Missing required field '{name}'.")
    for name in record:
        if name not in FEATURES:
            errors.append(f"Unexpected field '{name}' (allowed: {', '.join(FEATURES)}).")

    for name in NUMERIC_FEATURES:
        if name not in record:
            continue
        value = record[name]
        lo, hi = RANGES[name]
        if not _is_number(value):
            errors.append(f"'{name}' must be a number, got {type(value).__name__} {value!r}.")
        elif not math.isfinite(value):
            errors.append(f"'{name}' must be a finite number.")
        elif not lo <= value <= hi:
            errors.append(f"'{name}'={value} is outside the allowed range [{lo}, {hi}].")

    if "season" in record and record["season"] not in SEASONS:
        errors.append(f"'season' must be 'A' or 'B', got {record['season']!r}.")
    if "cooperative_support" in record:
        v = record["cooperative_support"]
        if isinstance(v, bool) or not isinstance(v, int):
            errors.append(f"'cooperative_support' must be the integer 0 or 1, got {v!r}.")
        elif v not in (0, 1):
            errors.append(f"'cooperative_support'={v} must be 0 or 1.")

    if errors:
        raise RecordValidationError(errors)
    return {k: record[k] for k in FEATURES}
