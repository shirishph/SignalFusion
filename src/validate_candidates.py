#!/usr/bin/env python3

import json
import sys
from pathlib import Path


REQUIRED_FIELDS = {
    "query_id",
    "candidate_id",
    "candidate_url",
    "image_url",
    "source",
    "discovery_method",
    "query",
    "domain",
}


def validate_record(record, line_number):
    errors = []

    if not isinstance(record, dict):
        return [f"Line {line_number}: record is not a JSON object"]

    missing = REQUIRED_FIELDS - record.keys()

    if missing:
        errors.append(
            f"Line {line_number}: missing fields: "
            + ", ".join(sorted(missing))
        )

    for field in REQUIRED_FIELDS:
        if field in record and not isinstance(record[field], str):
            errors.append(
                f"Line {line_number}: '{field}' must be a string"
            )

    return errors


def validate_file(path):
    errors = []
    records = 0

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(
                    f"Line {line_number}: invalid JSON: {exc}"
                )
                continue

            errors.extend(validate_record(record, line_number))
            records += 1

    print(f"Records: {records}")

    if errors:
        print("Validation: FAIL")
        for error in errors:
            print(f"  - {error}")
        return False

    print("Validation: PASS")
    return True


def main():
    if len(sys.argv) != 2:
        print(
            "Usage: python src/validate_candidates.py "
            "<candidates.jsonl>"
        )
        sys.exit(1)

    path = Path(sys.argv[1])

    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    if not validate_file(path):
        sys.exit(1)


if __name__ == "__main__":
    main()
