#!/usr/bin/env python3

import json
import re
import sys
from pathlib import Path


def normalize_text(text):
    """Normalize whitespace and remove surrounding noise."""
    return re.sub(r"\s+", " ", text).strip()


def filename_stem(filename):
    """Return filename without its extension."""
    return Path(filename).stem


def generate_queries(record):
    query_id = record["query_id"]
    filename = record.get("filename", "").strip()
    ocr_text = normalize_text(record.get("ocr_text", ""))

    queries = []

    # Filename with extension
    if filename:
        queries.append({
            "type": "filename",
            "query": f'"{filename}"',
        })

        # Filename without extension
        stem = filename_stem(filename)

        if stem and stem != filename:
            queries.append({
                "type": "filename_stem",
                "query": f'"{stem}"',
            })

    # OCR-based queries
    if ocr_text:
        queries.append({
            "type": "ocr_exact",
            "query": f'"{ocr_text}"',
        })

    return {
        "query_id": query_id,
        "queries": queries,
    }


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python src/generate_search_queries.py "
            "<input.jsonl> <output.jsonl>"
        )
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    records = 0
    queries_generated = 0

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

            result = generate_queries(record)

            outfile.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )

            records += 1
            queries_generated += len(result["queries"])

    print(f"Records processed: {records}")
    print(f"Queries generated: {queries_generated}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
