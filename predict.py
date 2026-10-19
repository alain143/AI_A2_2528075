"""Command-line prediction: python predict.py --json samples/valid.json"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.inference import ModelNotFoundError, Predictor
from src.validation import RecordValidationError


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Predict yield, risk and cluster for one farm record.")
    p.add_argument("--json", required=True, type=Path, help="path to a JSON file holding one record")
    args = p.parse_args(argv)
    try:
        record = json.loads(args.json.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(json.dumps({"error": f"File not found: {args.json}"}, indent=2), file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": f"Invalid JSON: {exc}"}, indent=2), file=sys.stderr)
        return 2
    try:
        result = Predictor().predict(record)
    except RecordValidationError as exc:
        print(json.dumps({"error": "invalid_input", "details": exc.errors}, indent=2), file=sys.stderr)
        return 2
    except ModelNotFoundError as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 3
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
