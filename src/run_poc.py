#!/usr/bin/env python3

import json
import subprocess
import sys
from pathlib import Path

from datetime import datetime
import requests
import json

from score_metadata import (
    platform_score,
    timestamp_score,
    metadata_score,
)

# Terminal colors
HOT_PINK = "\033[95m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RED = "\033[91m"
WHITE = "\033[97m"
RESET = "\033[0m"


def explain_image_score(image_score, phash_distance):
    if phash_distance == 0:
        return (
            f"{GREEN}The images are perceptually identical "
            f"at the hash level.{RESET}"
        )
    elif phash_distance <= 5:
        return (
            f"{GREEN}The images are extremely close "
            f"perceptually.{RESET}"
        )
    elif phash_distance <= 10:
        return (
            f"{GREEN}The images are very similar "
            f"perceptually.{RESET}"
        )
    elif phash_distance <= 20:
        return (
            f"{YELLOW}The images show moderate "
            f"perceptual similarity.{RESET}"
        )
    elif phash_distance <= 40:
        return (
            f"{YELLOW}The images show limited "
            f"perceptual similarity.{RESET}"
        )
    else:
        return (
            f"{RED}The images are perceptually "
            f"quite different.{RESET}"
        )


def explain_platform_score(platform_score_value):
    if platform_score_value == 1.0:
        return f"{GREEN}The platform matches.{RESET}"
    else:
        return f"{RED}The platform does not match.{RESET}"


def explain_timestamp_score(
    timestamp_score_value,
    difference_seconds,
):
    difference_hours = difference_seconds / 3600

    if difference_hours <= 1:
        return f"{GREEN}The timestamps are very close.{RESET}"
    elif difference_hours <= 6:
        return f"{GREEN}The timestamps are close.{RESET}"
    elif difference_hours <= 24:
        return (
            f"{YELLOW}The timestamps are within the comparison "
            f"window but substantially separated.{RESET}"
        )
    else:
        return (
            f"{RED}The timestamps are outside the "
            f"comparison window.{RESET}"
        )


def explain_overall(image_score, metadata_score_value):
    if image_score >= 0.9 and metadata_score_value >= 0.9:
        return (
            f"{YELLOW}The candidate has very strong image "
            f"similarity and strong metadata agreement.{RESET}"
        )

    if image_score >= 0.9 and metadata_score_value < 0.5:
        return (
            f"{YELLOW}The candidate is a very strong image match, "
            f"but the metadata provides weak supporting evidence.{RESET}"
        )

    if image_score < 0.7 and metadata_score_value >= 0.9:
        return (
            f"{YELLOW}The metadata strongly supports the candidate, "
            f"but the image similarity is comparatively weak.{RESET}"
        )

    if image_score >= 0.9:
        return (
            f"{YELLOW}The image strongly matches, while the metadata "
            f"provides only partial support.{RESET}"
        )

    if metadata_score_value >= 0.9:
        return (
            f"{YELLOW}The metadata strongly supports the candidate, "
            f"while the image provides weaker evidence.{RESET}"
        )

    return (
        f"{YELLOW}The image and metadata provide mixed "
        f"or limited evidence.{RESET}"
    )

def explain_metadata_score(
    metadata_score_value,
    platform_score_value,
    timestamp_score_value,
):
    if metadata_score_value >= 0.9:
        return (
            f"{GREEN}The metadata provides strong supporting "
            f"evidence: platform and timestamp are both closely aligned.{RESET}"
        )

    elif metadata_score_value >= 0.7:
        return (
            f"{YELLOW}The metadata provides good supporting "
            f"evidence, with some difference in platform or timestamp.{RESET}"
        )

    elif metadata_score_value >= 0.5:
        return (
            f"{YELLOW}The metadata provides partial supporting "
            f"evidence, with a noticeable mismatch in platform or time.{RESET}"
        )

    else:
        return (
            f"{RED}The metadata provides weak supporting "
            f"evidence because platform or timestamp alignment is poor.{RESET}"
        )


def fusion_score(image_score, metadata_score_value):
    return 0.5 * image_score + 0.5 * metadata_score_value


def explain_fusion_score(fusion_score_value):
    if fusion_score_value >= 0.9:
        return (
            f"{YELLOW}Strong agreement between image "
            f"and metadata signals.{RESET}"
        )
    elif fusion_score_value >= 0.7:
        return (
            f"{YELLOW}The combined signals provide "
            f"moderate supporting evidence.{RESET}"
        )
    elif fusion_score_value >= 0.5:
        return (
            f"{YELLOW}The combined signals provide "
            f"limited supporting evidence.{RESET}"
        )
    else:
        return (
            f"{RED}The combined signals provide weak "
            f"support for this candidate.{RESET}"
        )

def load_candidates(path, image_id):
    candidates = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)

            if record["query_id"] == image_id and record["image_url"]:
                candidates.append(record)

    return candidates


def hamming_distance(hex_a, hex_b):
    return (int(hex_a, 16) ^ int(hex_b, 16)).bit_count()


def terminal_link(path, label):
    absolute_path = Path(path).resolve()
    return (
        f"\033]8;;file://{absolute_path}\033\\"
        f"{label}"
        f"\033]8;;\033\\"
    )


def compare_fingerprints(original, candidate):
    phash_distance = hamming_distance(
        original["hashes"]["phash"],
        candidate["hashes"]["phash"],
    )

    image_score = max(0.0, 1.0 - phash_distance / 64.0)

    print(f"  {HOT_PINK}Image Score:    {image_score:.3f}{RESET}")
    print(f"      {explain_image_score(image_score, phash_distance)}")

    dhash_distance = hamming_distance(
        original["hashes"]["dhash"],
        candidate["hashes"]["dhash"],
    )

    sha256_match = (
        original["hashes"]["sha256"]
        == candidate["hashes"]["sha256"]
    )

    return sha256_match, phash_distance, dhash_distance, image_score


def download_image(url, output_path):
    print(f"Downloading: {url}")

    response = requests.get(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/153.0.0.0 Safari/537.36"
            )
        },
        timeout=15,
    )

    response.raise_for_status()

    output_path.write_bytes(response.content)

    print(f"Saved: {output_path}")
    print(f"Bytes: {len(response.content)}")


def run_fingerprint(image_dir, output_file):
    subprocess.run(
        [
            sys.executable,
            "src/extract_fingerprints.py",
            str(image_dir),
            str(output_file),
        ],
        check=True,
    )


def main():
    if len(sys.argv) != 2:
        print("Usage: python src/run_poc.py <image_id>")
        sys.exit(1)

    image_id = sys.argv[1]

    candidates_file = Path("data/candidates/candidates.jsonl")
    image_dir = Path("data/candidates/images")
    fingerprint_file = Path("data/fingerprints/candidates_e2e.jsonl")

    image_dir.mkdir(parents=True, exist_ok=True)

    with open("data/metadata/inputs.jsonl", encoding="utf-8") as f:
        metadata_input = json.loads(f.readline())

    metadata_candidates = {}

    with open("data/metadata/candidates.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                metadata_candidates[record["candidate_id"]] = record

    candidates = load_candidates(
        candidates_file,
        image_id,
    )

    if not candidates:
        print(f"No image candidates found for {image_id}")
        sys.exit(1)

    print(f"Original: {image_id}")
    print(f"Candidates with image URLs: {len(candidates)}")
    print()

    for index, candidate in enumerate(candidates, start=1):
        image_url = candidate["image_url"]

        output_path = (
            image_dir
            / f"{image_id}_candidate_{index}.jpg"
        )

        print(f"Candidate {index}")
        print(f"Page:  {candidate['candidate_url']}")
        print(f"Image: {image_url}")

        try:
            download_image(image_url, output_path)
        except requests.RequestException as exc:
            print(f"Download failed: {exc}")
            continue

        print()

        print("Fingerprinting downloaded candidates...")

        run_fingerprint(
            image_dir,
            fingerprint_file,
        )

        # Load original fingerprint
        with open(
            "data/fingerprints/originals.jsonl",
            encoding="utf-8",
        ) as f:
            originals = {
                json.loads(line)["image_id"]: json.loads(line)
                for line in f
                if line.strip()
            }

        # Load candidate fingerprints
        with open(
            fingerprint_file,
            encoding="utf-8",
        ) as f:
            candidate_fingerprints = {
                json.loads(line)["image_id"]: json.loads(line)
                for line in f
                if line.strip()
            }

        original = originals[image_id]

        print()
        print("COMPARISON")

        original_path = Path("data/originals") / f"{image_id}.jpg"

        for index, candidate in enumerate(candidates, start=1):
            candidate_id = f"{image_id}_candidate_{index}"
            candidate_fingerprint = candidate_fingerprints[candidate_id]

            sha256_match, phash_distance, dhash_distance, image_score = (
                compare_fingerprints(
                    original,
                    candidate_fingerprint,
                )
            )

            metadata_candidate = metadata_candidates[candidate["candidate_id"]]

            platform_score_value = platform_score(
                metadata_input["platform"],
                metadata_candidate["platform"],
            )

            timestamp_score_value = timestamp_score(
                metadata_input["timestamp"],
                metadata_candidate["timestamp"],
            )

            input_dt = datetime.fromisoformat(
                metadata_input["timestamp"]
            )

            candidate_dt = datetime.fromisoformat(
                metadata_candidate["timestamp"]
            )

            timestamp_difference_seconds = abs(
                (input_dt - candidate_dt).total_seconds()
            )



            metadata_score_value = metadata_score(
                metadata_input["platform"],
                metadata_input["timestamp"],
                metadata_candidate["platform"],
                metadata_candidate["timestamp"],
            )

            fusion_score_value = fusion_score(
                image_score,
                metadata_score_value,
            )

            print(
                f"  {HOT_PINK}Metadata Score: "
                f"      {metadata_score_value:.3f}{RESET}"
            )

            print(
                   "      " + explain_metadata_score(
                    metadata_score_value,
                    platform_score_value,
                    timestamp_score_value,
                )
            )

            print(
                f"  {HOT_PINK}Fusion Score:   "
                f"  {fusion_score_value:.3f}{RESET}"
            )

            print(
                f"      {explain_fusion_score(fusion_score_value)}"
            )

            print(f"  Platform: {metadata_input['platform']} → "
                  f"      {metadata_candidate['platform']}")
            print(f"      {explain_platform_score(platform_score_value)}")

            print(f"  Timestamp difference: "
                  f"      {timestamp_difference_seconds / 3600:.1f} hours")
            print(
                  f"      {explain_timestamp_score(timestamp_score_value, timestamp_difference_seconds)}"
            )

            print()

            print(f"Candidate {index}")

            candidate_path = (
                image_dir / f"{image_id}_candidate_{index}.jpg"
            )

            print(
                f"  Original:  "
                f"{terminal_link(original_path, 'open original')}"
            )

            print(
                f"  Candidate: "
                f"{terminal_link(candidate_path, 'open candidate')}"
            )


            print(f"  SHA-256 match:  {sha256_match}")
            print(f"  pHash distance: {phash_distance}")
            print(f"  dHash distance: {dhash_distance}")

            # Temporary integration rule.
            if sha256_match or phash_distance <= 5:
                decision = "MATCH"
            else:
                decision = "NO_MATCH"

            print(f"  Decision:       {decision}")

            evaluation_path = Path("data/evaluation/results.json")

            if evaluation_path.exists():
                with open(evaluation_path, "r", encoding="utf-8") as f:
                    evaluation = json.load(f)

                print()
                print("DETECTOR PERFORMANCE")
                print(f"  Accuracy:  {evaluation['accuracy']:.3f}")
                print(f"  Precision: {evaluation['precision']:.3f}")
                print(f"  Recall:    {evaluation['recall']:.3f}")
                print(f"  F1 Score:  {evaluation['f1']:.3f}")




        print()
        print("END-TO-END PIPELINE COMPLETE")

if __name__ == "__main__":
    main()
