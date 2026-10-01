#!/usr/bin/env python3

import json
import subprocess
import sys
from pathlib import Path


MAX_QUERIES = 3


def load_queries(path):
    """Load individual queries from search_queries.jsonl."""

    queries = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            for query in record.get("queries", []):
                queries.append({
                    "query_id": record["query_id"],
                    "query_type": query["type"],
                    "query": query["query"],
                })

    return queries


def main():
    if len(sys.argv) != 2:
        print(
            "Usage: python src/collect_candidates.py "
            "<search_queries.jsonl>"
        )
        sys.exit(1)

    input_path = Path(sys.argv[1])

    queries = load_queries(input_path)

    print(f"Queries available: {len(queries)}")
    print(f"Processing first {min(MAX_QUERIES, len(queries))} queries")
    print()

    for index, item in enumerate(queries[:MAX_QUERIES], start=1):
        print(
            f"[{index}/{min(MAX_QUERIES, len(queries))}] "
            f"{item['query_id']} / {item['query_type']}"
        )

        result = subprocess.run(
            [
                sys.executable,
                "src/discover_one.py",
                item["query_id"],
                item["query_type"],
                item["query"],
            ],
            check=False,
        )

        if result.returncode != 0:
            print("Stopping because the query failed.")
            sys.exit(result.returncode)

        print()


if __name__ == "__main__":
    main()
