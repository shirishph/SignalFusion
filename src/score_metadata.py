#!/usr/bin/env python3

from datetime import datetime, timezone

TIMESTAMP_TOLERANCE_SECONDS = 24 * 60 * 60


def platform_score(input_platform, candidate_platform):
    return 1.0 if input_platform.lower() == candidate_platform.lower() else 0.0


def timestamp_score(input_timestamp, candidate_timestamp):
    input_dt = datetime.fromisoformat(input_timestamp)
    candidate_dt = datetime.fromisoformat(candidate_timestamp)

    difference = abs((input_dt - candidate_dt).total_seconds())

    return max(
        0.0,
        1.0 - difference / TIMESTAMP_TOLERANCE_SECONDS,
    )


def metadata_score(
    input_platform,
    input_timestamp,
    candidate_platform,
    candidate_timestamp,
):
    p_score = platform_score(
        input_platform,
        candidate_platform,
    )

    t_score = timestamp_score(
        input_timestamp,
        candidate_timestamp,
    )

    return 0.5 * p_score + 0.5 * t_score

if __name__ == "__main__":
    import json

    with open("data/metadata/inputs.jsonl", encoding="utf-8") as f:
        input_record = json.loads(f.readline())

    with open("data/metadata/candidates.jsonl", encoding="utf-8") as f:
        candidate_record = json.loads(f.readline())

    p_score = platform_score(
        input_record["platform"],
        candidate_record["platform"],
    )

    t_score = timestamp_score(
        input_record["timestamp"],
        candidate_record["timestamp"],
    )

    score = metadata_score(
        input_record["platform"],
        input_record["timestamp"],
        candidate_record["platform"],
        candidate_record["timestamp"],
    )

    print(f"Query: {input_record['query_id']}")
    print(f"Candidate: {candidate_record['candidate_id']}")
    print(f"Platform score: {p_score:.3f}")
    print(f"Timestamp score: {t_score:.3f}")
    print(f"Metadata score: {score:.3f}")
