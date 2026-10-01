#!/usr/bin/env python3

import sys

from extract_image_urls import extract_image_urls


def main():
    if len(sys.argv) != 2:
        print(
            "Usage: python src/test_candidate_image_extraction.py "
            "<candidate_url>"
        )
        sys.exit(1)

    candidate_url = sys.argv[1]

    try:
        image_urls = extract_image_urls(candidate_url)
    except Exception as exc:
        print(f"Extraction failed: {exc}")
        sys.exit(1)

    print()
    print(f"Candidate URL: {candidate_url}")
    print(f"Image URLs found: {len(image_urls)}")

    for index, image_url in enumerate(image_urls, start=1):
        print(f"{index}. {image_url}")


if __name__ == "__main__":
    main()
