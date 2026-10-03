import base64
import html
import json
from pathlib import Path

import imagehash
import numpy as np


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

FINGERPRINTS_DIR = Path("data/fingerprints")
CANDIDATES_DIR = Path("candidates")
QUERY_IMAGE = Path("query_image/img_007.jpg")

HTML_REPORT = Path("report.html")

SIGNAL_WEIGHTS = {
    "phash": 0.25,
    "image_embedding": 0.25,
    "face_embedding": 0.25,
    "ocr": 0.25,
}

BAR_WIDTH = 10


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cosine_similarity(a, b):
    """Return cosine similarity in the range 0..1."""

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return None

    cosine = float(np.dot(a, b) / (norm_a * norm_b))

    # Convert [-1, 1] to [0, 1].
    similarity = (cosine + 1.0) / 2.0

    return max(0.0, min(1.0, similarity))


def phash_similarity(query_hash, candidate_hash):
    """Convert 64-bit pHash Hamming distance to 0..1 similarity."""

    query = imagehash.hex_to_hash(query_hash)
    candidate = imagehash.hex_to_hash(candidate_hash)

    distance = query - candidate

    return 1.0 - (distance / 64.0)


def text_similarity(query_text, candidate_text):
    """Simple token-based OCR similarity."""

    query_tokens = set(query_text.lower().split())
    candidate_tokens = set(candidate_text.lower().split())

    if not query_tokens and not candidate_tokens:
        return None

    if not query_tokens or not candidate_tokens:
        return 0.0

    intersection = query_tokens & candidate_tokens
    union = query_tokens | candidate_tokens

    return len(intersection) / len(union)


# ------------------------------------------------------------
# Signal comparison
# ------------------------------------------------------------

def compare_phash():

    query = load_json(
        FINGERPRINTS_DIR / "phash" / "query_image.json"
    )

    results = {}

    for path in sorted(
        (FINGERPRINTS_DIR / "phash").glob("c*.json")
    ):
        candidate = load_json(path)

        cid = candidate["cid"]

        results[cid] = phash_similarity(
            query["phash"],
            candidate["phash"],
        )

    return results


def compare_image_embeddings():

    query = load_json(
        FINGERPRINTS_DIR / "image_embedding" / "query_image.json"
    )

    results = {}

    for path in sorted(
        (FINGERPRINTS_DIR / "image_embedding").glob("c*.json")
    ):
        candidate = load_json(path)

        cid = candidate["cid"]

        results[cid] = cosine_similarity(
            query["embedding"],
            candidate["embedding"],
        )

    return results


def compare_face_embeddings():

    query = load_json(
        FINGERPRINTS_DIR / "face_embedding" / "query_image.json"
    )

    query_faces = query.get("faces", [])

    results = {}

    for path in sorted(
        (FINGERPRINTS_DIR / "face_embedding").glob("c*.json")
    ):
        candidate = load_json(path)

        cid = candidate["cid"]
        candidate_faces = candidate.get("faces", [])

        if not query_faces or not candidate_faces:
            results[cid] = None
            continue

        best_similarity = 0.0

        for query_face in query_faces:

            for candidate_face in candidate_faces:

                similarity = cosine_similarity(
                    query_face["embedding"],
                    candidate_face["embedding"],
                )

                if similarity is not None:
                    best_similarity = max(
                        best_similarity,
                        similarity,
                    )

        results[cid] = best_similarity

    return results


def compare_ocr():

    query = load_json(
        FINGERPRINTS_DIR / "ocr" / "query_image.json"
    )

    query_text = query.get("text", "")

    results = {}

    for path in sorted(
        (FINGERPRINTS_DIR / "ocr").glob("c*.json")
    ):
        candidate = load_json(path)

        cid = candidate["cid"]

        results[cid] = text_similarity(
            query_text,
            candidate.get("text", ""),
        )

    return results


# ------------------------------------------------------------
# Fusion
# ------------------------------------------------------------

def calculate_fusion_score(scores):

    weighted_sum = 0.0
    available_weight = 0.0

    for signal, weight in SIGNAL_WEIGHTS.items():

        score = scores.get(signal)

        if score is None:
            continue

        weighted_sum += score * weight
        available_weight += weight

    if available_weight == 0:
        return None

    return weighted_sum / available_weight


# ------------------------------------------------------------
# Interpretation
# ------------------------------------------------------------

def signal_strength(score):

    if score is None:
        return "unavailable"

    if score >= 0.90:
        return "very high similarity"

    if score >= 0.75:
        return "high similarity"

    if score >= 0.50:
        return "moderate similarity"

    if score >= 0.25:
        return "low similarity"

    return "very low similarity"


def interpret_signal(signal, score):

    if score is None:

        if signal == "phash":
            return "pHash comparison unavailable"

        if signal == "image_embedding":
            return "image embedding comparison unavailable"

        if signal == "face_embedding":
            return "face comparison unavailable"

        if signal == "ocr":
            return "no OCR comparison available"

    if signal == "phash":

        if score == 1.0:
            return "pHash is an exact match"

        return f"pHash shows {signal_strength(score)}"

    if signal == "image_embedding":
        return f"Image embedding shows {signal_strength(score)}"

    if signal == "face_embedding":
        return f"Face embedding shows {signal_strength(score)}"

    if signal == "ocr":

        if score == 0:
            return "no shared OCR text detected"

        return f"OCR shows {signal_strength(score)}"


def build_interpretation(scores):

    available = {
        signal: score
        for signal, score in scores.items()
        if score is not None
    }

    interpretations = []

    for signal in [
        "phash",
        "image_embedding",
        "face_embedding",
        "ocr",
    ]:

        interpretations.append(
            interpret_signal(
                signal,
                scores.get(signal),
            )
        )

    if not available:
        return (
            interpretations,
            "No comparable signals were available.",
        )

    strong_signals = [
        signal
        for signal, score in available.items()
        if score >= 0.75
    ]

    weak_signals = [
        signal
        for signal, score in available.items()
        if score < 0.50
    ]

    if len(strong_signals) >= 2:

        if weak_signals:
            summary = (
                "Multiple signals show high similarity, while "
                "other available signals are weaker."
            )
        else:
            summary = (
                "Multiple available signals show high similarity."
            )

    elif len(strong_signals) == 1:

        strong_signal = strong_signals[0]

        summary = (
            f"The strongest similarity is in "
            f"{strong_signal.replace('_', ' ')}, "
            "while the other available signals are lower."
        )

    else:

        summary = (
            "The available signals show mostly low-to-moderate "
            "similarity."
        )

    return interpretations, summary


# ------------------------------------------------------------
# Candidate metadata
# ------------------------------------------------------------

def load_candidate_metadata():

    candidates_file = CANDIDATES_DIR / "candidates.json"

    candidates = load_json(candidates_file)

    return {
        candidate["cid"]: candidate
        for candidate in candidates
    }


# ------------------------------------------------------------
# Terminal display
# ------------------------------------------------------------

def progress_bar(score):

    if score is None:
        return "N/A"

    filled = round(score * BAR_WIDTH)
    empty = BAR_WIDTH - filled

    return "█" * filled + "░" * empty


def format_score(score):

    if score is None:
        return "N/A"

    return f"{score:.2f}"


def print_report(results):

    print()
    print("SIGNALFUSION — CANDIDATE COMPARISON")
    print("=" * 82)
    print("Query: query_image")
    print()

    header = (
        f"{'CID':<8}"
        f"{'pHash':>8}"
        f"{'Image':>10}"
        f"{'Face':>10}"
        f"{'OCR':>10}"
        f"   {'Fusion Score'}"
    )

    print(header)
    print("-" * 82)

    for cid, scores in results.items():

        fusion_score = calculate_fusion_score(scores)

        row = (
            f"{cid:<8}"
            f"{format_score(scores['phash']):>8}"
            f"{format_score(scores['image_embedding']):>10}"
            f"{format_score(scores['face_embedding']):>10}"
            f"{format_score(scores['ocr']):>10}"
            f"   {progress_bar(fusion_score):<{BAR_WIDTH}}"
            f"  {format_score(fusion_score)}"
        )

        print(row)

    print("-" * 82)

    print()
    print("INTERPRETATION")
    print("-" * 82)

    for cid, scores in results.items():

        interpretations, summary = build_interpretation(scores)

        print()
        print(cid)

        for interpretation in interpretations:
            print(f"  • {interpretation}")

        print(f"  Interpretation: {summary}")

    print()


# ------------------------------------------------------------
# HTML utilities
# ------------------------------------------------------------

def image_to_data_uri(image_path):

    if not image_path.exists():
        return ""

    suffix = image_path.suffix.lower()

    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }

    mime_type = mime_types.get(suffix, "application/octet-stream")

    encoded = base64.b64encode(
        image_path.read_bytes()
    ).decode("ascii")

    return f"data:{mime_type};base64,{encoded}"


def html_score(score):

    if score is None:
        return '<span class="na">N/A</span>'

    return f"{score:.2f}"


def html_progress_bar(score):

    if score is None:
        return '<span class="na">N/A</span>'

    percentage = score * 100

    return f"""
    <div class="fusion">
        <div class="fusion-track">
            <div class="fusion-fill"
                 style="width: {percentage:.1f}%"></div>
        </div>
        <span class="fusion-value">{score:.2f}</span>
    </div>
    """


def create_html_report(results, candidate_metadata):

    query_uri = image_to_data_uri(QUERY_IMAGE)

    candidate_cards = []

    for cid, scores in results.items():

        metadata = candidate_metadata.get(cid, {})

        image_filename = metadata.get(
            "image_filename",
            "",
        )

        image_path = CANDIDATES_DIR / image_filename

        image_uri = image_to_data_uri(image_path)

        fusion_score = calculate_fusion_score(scores)

        interpretations, summary = build_interpretation(scores)

        interpretation_items = "".join(
            f"<li>{html.escape(item)}</li>"
            for item in interpretations
        )

        source_url = metadata.get("url", "")

        if source_url:
            source_html = (
                f'<a href="{html.escape(source_url)}" '
                f'target="_blank" rel="noopener">'
                f'View source'
                f'</a>'
            )
        else:
            source_html = ""

        candidate_cards.append(
            f"""
            <article class="candidate">

                <div class="candidate-header">
                    <div>
                        <div class="candidate-id">{html.escape(cid)}</div>
                        <div class="candidate-filename">
                            {html.escape(image_filename)}
                        </div>
                    </div>

                    <div class="fusion-header">
                        <div class="fusion-label">Fusion Score</div>
                        {html_progress_bar(fusion_score)}
                    </div>
                </div>

                <div class="candidate-body">

                    <div class="image-panel">
                        {
                            f'<img src="{image_uri}" '
                            f'alt="{html.escape(cid)}">'
                            if image_uri
                            else '<div class="missing-image">'
                                 'Image unavailable'
                                 '</div>'
                        }
                    </div>

                    <div class="details">

                        <div class="scores">
                            <div class="score">
                                <span>pHash</span>
                                <strong>
                                    {html_score(scores["phash"])}
                                </strong>
                            </div>

                            <div class="score">
                                <span>Image embedding</span>
                                <strong>
                                    {html_score(
                                        scores["image_embedding"]
                                    )}
                                </strong>
                            </div>

                            <div class="score">
                                <span>Face embedding</span>
                                <strong>
                                    {html_score(
                                        scores["face_embedding"]
                                    )}
                                </strong>
                            </div>

                            <div class="score">
                                <span>OCR</span>
                                <strong>
                                    {html_score(scores["ocr"])}
                                </strong>
                            </div>
                        </div>

                        <div class="interpretation">
                            <div class="section-label">
                                Interpretation
                            </div>

                            <ul>
                                {interpretation_items}
                            </ul>

                            <p class="summary">
                                {html.escape(summary)}
                            </p>
                        </div>

                        {source_html}

                    </div>

                </div>

            </article>
            """
        )

    candidates_html = "\n".join(candidate_cards)

    html_document = f"""
<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>SignalFusion Report</title>

<style>

    * {{
        box-sizing: border-box;
    }}

    body {{
        margin: 0;
        background: #f7f7f5;
        color: #222;
        font-family:
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            sans-serif;
    }}

    .container {{
        max-width: 1100px;
        margin: 0 auto;
        padding: 48px 28px 80px;
    }}

    header {{
        margin-bottom: 48px;
    }}

    h1 {{
        margin: 0 0 8px;
        font-size: 30px;
        font-weight: 600;
        letter-spacing: -0.5px;
    }}

    .subtitle {{
        color: #777;
        font-size: 14px;
    }}

    .query {{
        background: white;
        border: 1px solid #e5e5e2;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 36px;
    }}

    .section-label {{
        margin-bottom: 12px;
        color: #777;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
    }}

    .query img {{
        display: block;
        max-width: 320px;
        max-height: 360px;
        width: auto;
        height: auto;
        border-radius: 8px;
    }}

    .query-name {{
        margin-top: 12px;
        color: #777;
        font-size: 13px;
    }}

    .candidate {{
        background: white;
        border: 1px solid #e5e5e2;
        border-radius: 12px;
        margin-bottom: 20px;
        overflow: hidden;
    }}

    .candidate-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 30px;
        padding: 18px 20px;
        border-bottom: 1px solid #eeeeeb;
    }}

    .candidate-id {{
        font-size: 16px;
        font-weight: 600;
    }}

    .candidate-filename {{
        margin-top: 4px;
        color: #999;
        font-size: 12px;
        word-break: break-all;
    }}

    .fusion-header {{
        min-width: 250px;
    }}

    .fusion-label {{
        margin-bottom: 6px;
        color: #777;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }}

    .fusion {{
        display: flex;
        align-items: center;
        gap: 10px;
    }}

    .fusion-track {{
        width: 150px;
        height: 8px;
        background: #e9e9e6;
        border-radius: 20px;
        overflow: hidden;
    }}

    .fusion-fill {{
        height: 100%;
        background: #222;
        border-radius: 20px;
    }}

    .fusion-value {{
        min-width: 34px;
        font-size: 13px;
        font-weight: 600;
    }}

    .candidate-body {{
        display: grid;
        grid-template-columns: 260px 1fr;
        gap: 28px;
        padding: 20px;
    }}

    .image-panel {{
        display: flex;
        align-items: flex-start;
        justify-content: center;
        background: #f4f4f2;
        border-radius: 8px;
        overflow: hidden;
        min-height: 180px;
    }}

    .image-panel img {{
        display: block;
        width: 100%;
        height: 260px;
        object-fit: contain;
    }}

    .missing-image {{
        padding: 60px 20px;
        color: #999;
        font-size: 13px;
    }}

    .scores {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        margin-bottom: 28px;
    }}

    .score {{
        padding: 12px;
        background: #f7f7f5;
        border-radius: 8px;
    }}

    .score span {{
        display: block;
        margin-bottom: 7px;
        color: #888;
        font-size: 11px;
    }}

    .score strong {{
        font-size: 18px;
        font-weight: 600;
    }}

    .na {{
        color: #aaa;
        font-weight: 400;
    }}

    .interpretation {{
        padding-top: 2px;
    }}

    .interpretation ul {{
        margin: 0 0 12px;
        padding-left: 20px;
        color: #444;
        font-size: 14px;
        line-height: 1.7;
    }}

    .summary {{
        margin: 12px 0 16px;
        color: #555;
        font-size: 14px;
        line-height: 1.6;
    }}

    a {{
        color: #555;
        font-size: 13px;
    }}

    @media (max-width: 750px) {{

        .container {{
            padding: 28px 16px 60px;
        }}

        .candidate-header {{
            align-items: flex-start;
            flex-direction: column;
        }}

        .fusion-header {{
            width: 100%;
        }}

        .candidate-body {{
            grid-template-columns: 1fr;
        }}

        .image-panel img {{
            height: auto;
            max-height: 420px;
        }}

        .scores {{
            grid-template-columns: repeat(2, 1fr);
        }}

    }}

</style>

</head>

<body>

<div class="container">

    <header>
        <h1>SignalFusion</h1>
        <div class="subtitle">
            Multi-signal image comparison report
        </div>
    </header>


    <section class="query">

        <div class="section-label">
            Query image
        </div>

        {
            f'<img src="{query_uri}" alt="Query image">'
            if query_uri
            else '<div class="missing-image">'
                 'Query image unavailable'
                 '</div>'
        }

        <div class="query-name">
            {html.escape(QUERY_IMAGE.name)}
        </div>

    </section>


    <section>

        <div class="section-label">
            Candidates
        </div>

        {candidates_html}

    </section>

</div>

</body>

</html>
"""

    HTML_REPORT.write_text(
        html_document,
        encoding="utf-8",
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    phash_scores = compare_phash()
    image_scores = compare_image_embeddings()
    face_scores = compare_face_embeddings()
    ocr_scores = compare_ocr()

    candidate_ids = sorted(phash_scores.keys())

    results = {}

    for cid in candidate_ids:

        results[cid] = {
            "phash": phash_scores.get(cid),
            "image_embedding": image_scores.get(cid),
            "face_embedding": face_scores.get(cid),
            "ocr": ocr_scores.get(cid),
        }

    # --------------------------------------------------------
    # Calculate Fusion Score and sort highest first.
    # --------------------------------------------------------

    scored_results = []

    for cid, scores in results.items():

        fusion_score = calculate_fusion_score(scores)

        scored_results.append(
            (
                cid,
                scores,
                fusion_score,
            )
        )

    scored_results.sort(
        key=lambda item: (
            item[2] is not None,
            item[2] if item[2] is not None else -1,
        ),
        reverse=True,
    )

    sorted_results = {
        cid: scores
        for cid, scores, _ in scored_results
    }

    # --------------------------------------------------------
    # Terminal report
    # --------------------------------------------------------

    print_report(sorted_results)

    # --------------------------------------------------------
    # HTML report
    # --------------------------------------------------------

    candidate_metadata = load_candidate_metadata()

    create_html_report(
        sorted_results,
        candidate_metadata,
    )

    print(f"HTML report: {HTML_REPORT}")


if __name__ == "__main__":
    main()
