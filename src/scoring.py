from __future__ import annotations

from math import sqrt
from typing import Any


def exact_hash_similarity(
    query: dict[str, Any],
    candidate: dict[str, Any],
) -> float:
    """Compare SHA-256 hashes."""

    q = query.get("hashes", {}).get("sha256")
    c = candidate.get("hashes", {}).get("sha256")

    if not q or not c:
        return 0.0

    return 1.0 if q == c else 0.0


def perceptual_hash_similarity(
    query: dict[str, Any],
    candidate: dict[str, Any],
) -> float:
    """Compare pHash and dHash using normalized Hamming similarity."""

    q_hashes = query.get("hashes", {})
    c_hashes = candidate.get("hashes", {})

    similarities = []

    for hash_name in ("phash", "dhash"):
        q = q_hashes.get(hash_name)
        c = c_hashes.get(hash_name)

        if not q or not c:
            continue

        q_int = int(q, 16)
        c_int = int(c, 16)

        distance = (q_int ^ c_int).bit_count()

        # 64-bit perceptual hashes
        similarity = 1.0 - (distance / 64.0)

        similarities.append(similarity)

    if not similarities:
        return 0.0

    return sum(similarities) / len(similarities)


def cosine_similarity(
    query_vector: list[float],
    candidate_vector: list[float],
) -> float:
    """Calculate cosine similarity and normalize to 0..1."""

    if not query_vector or not candidate_vector:
        return 0.0

    if len(query_vector) != len(candidate_vector):
        return 0.0

    dot = sum(
        a * b
        for a, b in zip(query_vector, candidate_vector)
    )

    q_norm = sqrt(sum(a * a for a in query_vector))
    c_norm = sqrt(sum(b * b for b in candidate_vector))

    if q_norm == 0 or c_norm == 0:
        return 0.0

    cosine = dot / (q_norm * c_norm)

    # [-1, 1] → [0, 1]
    return (cosine + 1.0) / 2.0


def face_embedding_similarity(
    query: dict[str, Any],
    candidate: dict[str, Any],
) -> float:
    """Compare the first detected face in each image."""

    q_faces = query.get("faces", [])
    c_faces = candidate.get("faces", [])

    if not q_faces or not c_faces:
        return 0.0

    q_embedding = (
        q_faces[0]
        .get("embedding", {})
        .get("vector")
    )

    c_embedding = (
        c_faces[0]
        .get("embedding", {})
        .get("vector")
    )

    if not q_embedding or not c_embedding:
        return 0.0

    return cosine_similarity(q_embedding, c_embedding)


def ocr_similarity(
    query: dict[str, Any],
    candidate: dict[str, Any],
) -> float:
    """Compare OCR text using word-level Jaccard similarity."""

    q_text = (
        query.get("ocr", {})
        .get("text", "")
        .strip()
        .lower()
    )

    c_text = (
        candidate.get("ocr", {})
        .get("text", "")
        .strip()
        .lower()
    )

    if not q_text or not c_text:
        return 0.0

    if q_text == c_text:
        return 1.0

    q_words = set(q_text.split())
    c_words = set(c_text.split())

    union = q_words | c_words

    if not union:
        return 0.0

    return len(q_words & c_words) / len(union)


def filename_similarity(
    query: dict[str, Any],
    candidate: dict[str, Any],
) -> float:
    """Compare filename clues."""

    q_name = (
        query.get("source", {})
        .get("filename", "")
        .lower()
    )

    c_name = (
        candidate.get("source", {})
        .get("filename", "")
        .lower()
    )

    if not q_name or not c_name:
        return 0.0

    if q_name == c_name:
        return 1.0

    q_parts = set(
        q_name.replace("_", " ")
        .replace("-", " ")
        .split()
    )

    c_parts = set(
        c_name.replace("_", " ")
        .replace("-", " ")
        .split()
    )

    union = q_parts | c_parts

    if not union:
        return 0.0

    return len(q_parts & c_parts) / len(union)


def compare(
    query: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Compare two fingerprint records using all signals."""

    return {
        "query_id": query.get("image_id"),
        "candidate_id": candidate.get("image_id"),

        "signals": {
            "exact_hash": exact_hash_similarity(
                query, candidate
            ),

            "perceptual_hash": perceptual_hash_similarity(
                query, candidate
            ),

            "face_embedding": face_embedding_similarity(
                query, candidate
            ),

            "ocr": ocr_similarity(
                query, candidate
            ),

            "filename": filename_similarity(
                query, candidate
            ),
        },
    }
