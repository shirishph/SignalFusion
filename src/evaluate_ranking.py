from collections import defaultdict
from pathlib import Path

from compare_candidates import (
    load_fingerprints,
    embedding_similarity,
    face_similarity,
    ocr_similarity,
    combined_score,
    compare_hashes,
)


BASE_DIR = Path(__file__).resolve().parent.parent

ORIGINALS_FILE = BASE_DIR / "data/fingerprints/originals.jsonl"
VARIANTS_FILE = BASE_DIR / "data/fingerprints/variants.jsonl"


def rank_candidates(query, variants):
    results = []

    for candidate_id, candidate in variants.items():
        hashes = compare_hashes(query, candidate)

        embedding = embedding_similarity(query, candidate)
        face = face_similarity(query, candidate)
        ocr = ocr_similarity(query, candidate)

        score = combined_score(
            hashes["exact_hash"],
            hashes["phash_similarity"],
            embedding,
            face,
            ocr,
            None,
        )

        results.append({
            "candidate_id": candidate_id,
            "score": score,
            "exact_hash": hashes["exact_hash"],
            "phash": hashes["phash_similarity"],
            "embedding": embedding,
            "face": face,
            "ocr": ocr,
        })

    return results


def recall_at_k(ranked, positive_ids, k):
    top_k = ranked[:k]
    retrieved = {item["candidate_id"] for item in top_k}

    return 1 if retrieved.intersection(positive_ids) else 0


def evaluate_signal(results, signal, positive_ids):
    ranked = sorted(
        results,
        key=lambda x: (
            -1.0 if x[signal] is None else x[signal]
        ),
        reverse=True,
    )

    return {
        "r1": recall_at_k(ranked, positive_ids, 1),
        "r5": recall_at_k(ranked, positive_ids, 5),
        "r10": recall_at_k(ranked, positive_ids, 10),
    }


def main():
    originals = load_fingerprints(ORIGINALS_FILE)
    variants = load_fingerprints(VARIANTS_FILE)

    print(f"Originals: {len(originals)}")
    print(f"Variants: {len(variants)}")

    results_by_query = {}

    for query_id, query in originals.items():

        # Positive candidates are variants whose ID starts
        # with the query's original ID.
        positive_ids = {
            candidate_id
            for candidate_id in variants
            if candidate_id.startswith(query_id + "_")
        }

        results = rank_candidates(query, variants)

        results_by_query[query_id] = {
            "results": results,
            "positives": positive_ids,
        }

    signals = [
        ("multi_signal", "score"),
        ("exact_hash", "exact_hash"),
        ("phash", "phash"),
        ("image_embedding", "embedding"),
        ("face_embedding", "face"),
        ("ocr", "ocr"),
    ]

    print("\n=== Recall ===")
    print(
        f"{'Signal':<20}"
        f"{'Recall@1':>12}"
        f"{'Recall@5':>12}"
        f"{'Recall@10':>12}"
    )

    for signal_name, signal_key in signals:
        totals = {
            "r1": 0,
            "r5": 0,
            "r10": 0,
        }

        for query_data in results_by_query.values():
            metrics = evaluate_signal(
                query_data["results"],
                signal_key,
                query_data["positives"],
            )

            totals["r1"] += metrics["r1"]
            totals["r5"] += metrics["r5"]
            totals["r10"] += metrics["r10"]

        n = len(results_by_query)

        print(
            f"{signal_name:<20}"
            f"{totals['r1'] / n:>12.3f}"
            f"{totals['r5'] / n:>12.3f}"
            f"{totals['r10'] / n:>12.3f}"
        )


if __name__ == "__main__":
    main()
