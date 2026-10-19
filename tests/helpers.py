"""Shared helpers: make sure trained artifacts exist before inference tests run."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STUDENT_ID = "ug2528075@ines.ac.rw"
SEED = 8075


def ensure_trained() -> None:
    """Run the full pipeline once if models are missing (keeps tests self-contained)."""
    if not (ROOT / "models" / "metadata.json").exists():
        from src.pipeline import run

        run(STUDENT_ID, SEED, verbose=False)


def valid_record() -> dict:
    return json.loads((ROOT / "samples" / "valid.json").read_text(encoding="utf-8"))
