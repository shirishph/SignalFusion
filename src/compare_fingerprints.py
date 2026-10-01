#!/usr/bin/env python3

import json
import sys


def load_records(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def find_record(records, image_id):
    for record in records:
        if record["image_id"] == image_id:
            return record

    raise ValueError(f"Image ID not found: {image_id}")


def hamming_distance(hex_a, hex_b):
    value_a = int(hex_a, 16)
    value_b = int(hex_b, 16)

    return (value_a ^ value_b).bit_count()


def compare(original, candidate):
    return {
        "sha256_match": (
            original["hashes"]["sha256"]
            == candidate["hashes"]["sha256"]
        ),
        "phash_distance": hamming_distance(
            original["hashes"]["phash"],
            candidate["hashes"]["phash"],
        ),
        "dhash_distance": hamming_distance(
            original["hashes"]["dhash"],
            candidate["hashes"]["dhash"],
        ),
    }


def main():
    if len(sys.argv) != 4:
        print(
            "Usage: python src/compare_fingerprints.py "
            "<original.jsonl> <candidate.jsonl> <image_id>"
        )
        sys.exit(1)

    original_records = load_records(sys.argv[1])
    candidate_records = load_records(sys.argv[2])
    image_id = sys.argv[3]

    original = find_record(original_records, image_id)

    candidate = find_record(
        candidate_records,
        f"{image_id}_pinterest",
    )

    results = compare(original, candidate)

    print(f"Image ID:            {image_id}")
    print(f"SHA-256 match:       {results['sha256_match']}")
    print(f"pHash distance:      {results['phash_distance']}")
    print(f"dHash distance:      {results['dhash_distance']}")


if __name__ == "__main__":
    main()
