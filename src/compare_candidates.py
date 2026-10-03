from collections import defaultdict
import json
from pathlib import Path

FINGERPRINTS_FILE = Path("data/fingerprints/candidates.jsonl")
ORIGINALS_FILE = Path("data/fingerprints/originals.jsonl")
CANDIDATES_FILE = Path("data/candidates/candidates.jsonl")
IMAGES_DIR = Path("data/candidates/images")


def load_fingerprints(path: Path) -> dict:
    fingerprints = {}

    with path.open() as f:
        for line in f:
            record = json.loads(line)
            fingerprints[record["image_id"]] = record

    return fingerprints


def load_candidates(path: Path) -> list:
    candidates = []

    with path.open() as f:
        for line in f:
            candidates.append(json.loads(line))

    return candidates


def find_image_id(candidate: dict) -> str | None:
    query_id = candidate["query_id"]

    if candidate["source"] == "manual":
        path = IMAGES_DIR / f"{query_id}_pinterest.jpg"
    else:
        path = IMAGES_DIR / f"{query_id}_candidate_1.jpg"

    if path.exists():
        return path.stem

    return None


def hamming_distance_hex(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def phash_similarity(a: str, b: str) -> float:
    distance = hamming_distance_hex(a, b)
    max_distance = len(a) * 4
    return 1.0 - (distance / max_distance)


def compare_hashes(query: dict, candidate: dict) -> dict:
    query_hashes = query["hashes"]
    candidate_hashes = candidate["hashes"]

    exact_match = (
        query_hashes["sha256"] == candidate_hashes["sha256"]
    )

    phash_score = phash_similarity(
        query_hashes["phash"],
        candidate_hashes["phash"],
    )

    return {
        "exact_hash": 1.0 if exact_match else 0.0,
        "phash_similarity": phash_score,
    }


def embedding_similarity(query: dict, candidate: dict) -> float:
    query_vector = query["image_embedding"]["vector"]
    candidate_vector = candidate["image_embedding"]["vector"]

    return sum(
        q * c
        for q, c in zip(query_vector, candidate_vector)
    )

def face_similarity(query: dict, candidate: dict) -> float | None:
    query_faces = query.get("faces", [])
    candidate_faces = candidate.get("faces", [])

    if not query_faces or not candidate_faces:
        return None

    query_vector = query_faces[0]["embedding"]["vector"]
    candidate_vector = candidate_faces[0]["embedding"]["vector"]

    query_norm = sum(x * x for x in query_vector) ** 0.5
    candidate_norm = sum(x * x for x in candidate_vector) ** 0.5

    if query_norm == 0 or candidate_norm == 0:
        return None

    return sum(
        q * c
        for q, c in zip(query_vector, candidate_vector)
    ) / (query_norm * candidate_norm)


def ocr_similarity(query: dict, candidate: dict) -> float | None:
    query_text = query.get("ocr", {}).get("text", "").strip().lower()
    candidate_text = candidate.get("ocr", {}).get("text", "").strip().lower()

    if not query_text or not candidate_text:
        return None

    if query_text == candidate_text:
        return 1.0

    query_words = set(query_text.split())
    candidate_words = set(candidate_text.split())

    if not query_words or not candidate_words:
        return None

    intersection = query_words & candidate_words
    union = query_words | candidate_words

    return len(intersection) / len(union)

def clue_similarity(query: dict, candidate: dict) -> float | None:
    query_filename = query["source"]["filename"].lower()

    candidate_text = " ".join([
        candidate.get("query", ""),
        candidate.get("title", ""),
        candidate.get("snippet", ""),
    ]).lower()

    if not candidate_text:
        return None

    filename_stem = Path(query_filename).stem.lower()

    if query_filename in candidate_text:
        return 1.0

    if filename_stem in candidate_text:
        return 1.0

    return 0.0

def combined_score(
    exact_score: float,
    phash_score: float,
    embedding_score: float,
    face_score: float | None,
    ocr_score: float | None,
    clue_score: float | None,
) -> float:

    weighted_scores = [
        (exact_score, 0.25),
        (phash_score, 0.20),
        (embedding_score, 0.20),
    ]

    if face_score is not None:
        weighted_scores.append((face_score, 0.15))

    if ocr_score is not None:
        weighted_scores.append((ocr_score, 0.10))

    if clue_score is not None:
        weighted_scores.append((clue_score, 0.10))

    numerator = sum(
        score * weight
        for score, weight in weighted_scores
    )

    denominator = sum(
        weight
        for _, weight in weighted_scores
    )

    return numerator / denominator

def explain_match(result: dict) -> str:
    reasons = []

    if result["exact_hash"] == 1.0:
        reasons.append("exact hash match")

    if result["phash"] >= 0.90:
        reasons.append(f"strong perceptual similarity ({result['phash']:.3f})")
    elif result["phash"] >= 0.75:
        reasons.append(f"moderate perceptual similarity ({result['phash']:.3f})")

    if result["embedding"] >= 0.90:
        reasons.append(f"strong image similarity ({result['embedding']:.3f})")
    elif result["embedding"] >= 0.75:
        reasons.append(f"moderate image similarity ({result['embedding']:.3f})")

    if result["face"] is not None:
        if result["face"] >= 0.90:
            reasons.append(f"strong face similarity ({result['face']:.3f})")
        elif result["face"] >= 0.75:
            reasons.append(f"moderate face similarity ({result['face']:.3f})")

    if result["ocr"] is not None:
        reasons.append(f"OCR similarity ({result['ocr']:.3f})")

    if result["clue"] is not None:
        if result["clue"] > 0:
            reasons.append("filename/search clue match")
        else:
            reasons.append("no filename/search clue match")

    if not reasons:
        return "No strong supporting signals"

    return "; ".join(reasons)



if __name__ == "__main__":
    original_fingerprints = load_fingerprints(ORIGINALS_FILE)
    candidate_fingerprints = load_fingerprints(FINGERPRINTS_FILE)
    candidates = load_candidates(CANDIDATES_FILE)

    print(f"Original fingerprints: {len(original_fingerprints)}")
    print(f"Candidate fingerprints: {len(candidate_fingerprints)}")
    print(f"Candidates: {len(candidates)}")

    results = []

    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        query_id = candidate["query_id"]

        image_id = find_image_id(candidate)

        if image_id is None:
            print(f"{candidate_id}: no local image")
            continue

        if image_id not in candidate_fingerprints:
            print(f"{candidate_id} -> {image_id}: fingerprint NOT FOUND")
            continue

        query = original_fingerprints.get(query_id)

        if query is None:
            print(f"{candidate_id}: original fingerprint NOT FOUND")
            continue

        candidate_fp = candidate_fingerprints[image_id]

        scores = compare_hashes(query, candidate_fp)
        embedding_score = embedding_similarity(query, candidate_fp)
        face_score = face_similarity(query, candidate_fp)
        ocr_score = ocr_similarity(query, candidate_fp)
        clue_score = clue_similarity(query, candidate)

        final_score = combined_score(
            scores["exact_hash"],
            scores["phash_similarity"],
            embedding_score,
            face_score,
            ocr_score,
            clue_score,
        )

        face_display = (
            "N/A" if face_score is None else f"{face_score:.3f}"
        )

        ocr_display = (
            "N/A" if ocr_score is None else f"{ocr_score:.3f}"
        )

        clue_display = (
            "N/A" if clue_score is None else f"{clue_score:.3f}"
        )

        """
        print(
            f"{candidate_id} | "
            f"exact={scores['exact_hash']:.0f} | "
            f"phash={scores['phash_similarity']:.3f} | "
            f"embedding={embedding_score:.3f} | "
            f"face={face_display} | "
            f"ocr={ocr_display} | "
            f"clue={clue_display} | "
            f"score={final_score:.3f}"
        )
        """

        results.append({
            "query_id": query_id,
            "candidate_id": candidate_id,
            "score": final_score,
            "exact_hash": scores["exact_hash"],
            "phash": scores["phash_similarity"],
            "embedding": embedding_score,
            "face": face_score,
            "ocr": ocr_score,
            "clue": clue_score,
        })

    ranked = defaultdict(list)

    for result in results:
        ranked[result["query_id"]].append(result)

    for query_id, candidates_for_query in ranked.items():
        candidates_for_query.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        print(f"\n=== {query_id} ===")

        for rank, result in enumerate(candidates_for_query, start=1):
            print(
                f"{rank}. {result['candidate_id']} "
                f"score={result['score']:.3f}"
            )
            print(f"   Why: {explain_match(result)}")


