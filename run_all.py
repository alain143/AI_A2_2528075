"""Single entry point: python run_all.py --student-id ug2528075@ines.ac.rw --seed 8075"""
from __future__ import annotations

import argparse
import sys

from src.pipeline import DEFAULT_NAME, run
from src.validation import ValidationError


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Recreate data, train all models and regenerate every artifact.")
    p.add_argument("--student-id", required=True, help="institutional username, e.g. ug2528075@ines.ac.rw")
    p.add_argument("--seed", required=True, type=int, help="LLLL = last four digits of the registration number")
    p.add_argument("--student-name", default=DEFAULT_NAME, help="full name shown in model info")
    args = p.parse_args(argv)
    try:
        run(args.student_id, args.seed, args.student_name)
    except ValidationError as exc:
        print("ERROR:", *exc.errors, sep="\n  ", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
