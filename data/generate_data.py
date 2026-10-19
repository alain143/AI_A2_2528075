"""Individual synthetic dataset generator for the Musanze Harvest Intelligence Service.

Usage (from the project root)::

    python data/generate_data.py --seed 8075 --out data/AI_A2_2528075.csv

Random draw order (all from ``numpy.random.default_rng(seed)``):
  1. rainfall, temperature, soil moisture, fertilizer, field size (uniform)
  2. season (A/B), cooperative support (0/1)
  3. yield noise N(0, 0.80)
  4. m unique row positions -> soil_moisture_pct set to missing
  5. m unique row positions -> fertilizer_kg set to missing (independent draw)
  6. 3 unique row positions -> copied and appended as exact duplicates
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

N_DUPLICATES = 3


def base_row_count(seed: int) -> int:
    """Base rows = 240 + (LLLL mod 41), i.e. 240..280."""
    return 240 + (int(seed) % 41)


def missing_count(n_base: int) -> int:
    """m = round(0.05 * base row count)."""
    return int(round(0.05 * n_base))


def generate_dataframe(seed: int) -> pd.DataFrame:
    """Create the raw dataset (with deliberate missing values and duplicates)."""
    rng = np.random.default_rng(int(seed))
    n = base_row_count(seed)

    rainfall = rng.uniform(600, 1400, n)
    temp = rng.uniform(12, 26, n)
    soil = rng.uniform(20, 80, n)
    fert = rng.uniform(0, 250, n)
    field = rng.uniform(0.2, 4.0, n)
    season = rng.choice(np.array(["A", "B"]), size=n)
    support = rng.integers(0, 2, size=n)
    noise = rng.normal(0.0, 0.80, n)

    season_effect = np.where(season == "A", 0.50, -0.20)
    yield_tons = (
        0.004 * rainfall
        + 0.06 * soil
        + 0.02 * fert
        + 1.30 * field
        - 0.25 * np.abs(temp - 18)
        + 0.70 * support
        + season_effect
        + noise
    )
    yield_tons = np.clip(yield_tons, 0.50, None)

    yield_per_ha = yield_tons / field
    high_risk = (yield_per_ha < np.median(yield_per_ha)).astype(int)

    df = pd.DataFrame(
        {
            "rainfall_mm": rainfall,
            "avg_temp_c": temp,
            "soil_moisture_pct": soil,
            "fertilizer_kg": fert,
            "field_size_ha": field,
            "season": season,
            "cooperative_support": support,
            "yield_tons": yield_tons,
            "high_risk": high_risk,
        }
    )

    m = missing_count(n)
    soil_pos = rng.choice(n, size=m, replace=False)
    fert_pos = rng.choice(n, size=m, replace=False)
    df.loc[soil_pos, "soil_moisture_pct"] = np.nan
    df.loc[fert_pos, "fertilizer_kg"] = np.nan

    dup_pos = rng.choice(n, size=N_DUPLICATES, replace=False)
    df = pd.concat([df, df.iloc[dup_pos]], ignore_index=True)
    return df


def write_csv(df: pd.DataFrame, path: Path) -> str:
    """Write the CSV (4 decimals, LF line endings) and return its SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.4f", lineterminator="\n")
    return sha256_of_file(path)


def sha256_of_file(path: Path) -> str:
    """SHA-256 of the exact bytes of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_fingerprint(path: Path, csv_path: Path, sha: str, n_rows: int, seed: int) -> None:
    """Write artifacts/data_fingerprint.txt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        f"verification_code: AI-A2-{int(seed):04d}\n"
        f"file: {Path(csv_path).name}\n"
        f"rows: {n_rows}\n"
        f"sha256: {sha}\n"
    )
    path.write_text(text, encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, required=True, help="LLLL (last four digits)")
    parser.add_argument("--out", type=Path, required=True, help="output CSV path")
    parser.add_argument("--fingerprint", type=Path, default=None, help="fingerprint txt path")
    args = parser.parse_args(argv)

    df = generate_dataframe(args.seed)
    sha = write_csv(df, args.out)
    fp = args.fingerprint or Path(args.out).resolve().parents[1] / "artifacts" / "data_fingerprint.txt"
    write_fingerprint(fp, args.out, sha, len(df), args.seed)
    print(f"AI-A2-{args.seed:04d}  rows={len(df)}  sha256={sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
