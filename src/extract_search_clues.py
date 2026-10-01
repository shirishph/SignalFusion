#!/usr/bin/env python3

import json
import sys
from pathlib import Path


def extract_clues(record):
    image_id = record["image_id"]

    source = record.get("source", {})
    metadata = record.get("metadata", {})
    ocr = record.get("ocr", {})

    filename = source.get("filename") or metadata.get("filename", "")
    ocr_text = ocr.get("text", "").strip()

    return {
        "query_id": image_id,
        "filename": filename,
        "ocr_text": ocr_text,
        "has_text": bool(ocr_text),
        "width": metadata.get("width"),
        "height": metadata.get("height"),
        "format": metadata.get("format"),
    }


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python src/extract_search_clues.py "
            "<input.jsonl> <output.jsonl>"
        )
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    records = 0

    with (
        input_path.open("r", encoding="utf-8") as infile,
        output_path.open("w", encoding="utf-8") as outfile,
    ):
        for line_number, line in enumerate(infile, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"ERROR: Invalid JSON on line {line_number}: {exc}",
                    file=sys.stderr,
                )
                sys.exit(1)

            clues = extract_clues(record)

            outfile.write(json.dumps(clues) + "\n")
            records += 1

    print(f"Records processed: {records}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
