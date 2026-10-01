#!/usr/bin/env python3

import csv
import json
import sys
from collections import Counter
import json
from pathlib import Path

PHASH_THRESHOLD = 5
EMBEDDING_THRESHOLD = 0.88

def load_jsonl(path):
    records = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                records[record["image_id"]] = record

    return records


def hamming_distance(hex_a, hex_b):
    return (int(hex_a, 16) ^ int(hex_b, 16)).bit_count()


def cosine_similarity(vector_a, vector_b):
    return sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )


def get_embedding_similarity(query, candidate):
    query_vector = query["image_embedding"]["vector"]
    candidate_vector = candidate["image_embedding"]["vector"]

    return cosine_similarity(
        query_vector,
        candidate_vector,
    )


def is_match(query, candidate):
    sha_match = (
        query["hashes"]["sha256"]
        == candidate["hashes"]["sha256"]
    )

    phash_distance = hamming_distance(
        query["hashes"]["phash"],
        candidate["hashes"]["phash"],
    )

    embedding_similarity = get_embedding_similarity(
        query,
        candidate,
    )

    return (
        sha_match
        or phash_distance <= PHASH_THRESHOLD
        or embedding_similarity >= EMBEDDING_THRESHOLD
    )

def main():
    if len(sys.argv) != 4:
        print(
            "Usage: python src/evaluate_detector.py "
            "<originals.jsonl> <variants.jsonl> <pairs.csv>"
        )
        sys.exit(1)

    originals = load_jsonl(sys.argv[1])
    variants = load_jsonl(sys.argv[2])

    fingerprints = {}
    fingerprints.update(originals)
    fingerprints.update(variants)

    results = []

    with open(sys.argv[3], newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            query = fingerprints[row["query_id"]]
            candidate = fingerprints[row["candidate_id"]]

            predicted_match = is_match(query, candidate)

            embedding_similarity = get_embedding_similarity(
                query,
                candidate,
            )

            expected_match = row["relationship"] not in {
                "EASY_NEGATIVE",
                "HARD_NEGATIVE",
            }

            results.append({
                "relationship": row["relationship"],
                "expected_match": expected_match,
                "predicted_match": predicted_match,
                "embedding_similarity": embedding_similarity,
            })

    tp = sum(
        r["expected_match"] and r["predicted_match"]
        for r in results
    )

    tn = sum(
        not r["expected_match"] and not r["predicted_match"]
        for r in results
    )

    fp = sum(
        not r["expected_match"] and r["predicted_match"]
        for r in results
    )

    fn = sum(
        r["expected_match"] and not r["predicted_match"]
        for r in results
    )

    total = len(results)

    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0
    )

    print(f"Pairs evaluated: {total}")
    print(f"pHash threshold: {PHASH_THRESHOLD}")
    print(f"Embedding threshold: {EMBEDDING_THRESHOLD}")
    print()

    print(f"TP: {tp}")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print()

    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print()

    print("Embedding similarity by relationship:")
    print()

    relationships = sorted(
        {r["relationship"] for r in results}
    )

    for relationship in relationships:
        similarities = [
            r["embedding_similarity"]
            for r in results
            if r["relationship"] == relationship
        ]

        print(
            f"{relationship}: "
            f"min={min(similarities):.4f}, "
            f"max={max(similarities):.4f}, "
            f"avg={sum(similarities) / len(similarities):.4f}"
        )

    print()

    counts = Counter(
        (r["relationship"], r["predicted_match"])
        for r in results
    )

    for relationship in relationships:
        match_count = counts[(relationship, True)]
        no_match_count = counts[(relationship, False)]

        print(
            f"{relationship}: "
            f"MATCH={match_count}, "
            f"NO_MATCH={no_match_count}"
        )

    results_data = {
        "pairs_evaluated": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    output_path = Path("data/evaluation/results.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    print(f"F1:        {f1:.3f}")
    print(f"Results written to: {output_path}")

if __name__ == "__main__":
    main()
