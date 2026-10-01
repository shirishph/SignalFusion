#!/usr/bin/env python3

import sys
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/153.0.0.0 Safari/537.36"
    )
}


def add_image_url(image_urls, page_url, image_url):
    """Add a normalized image URL if it is new."""

    if not image_url:
        return

    absolute_url = urljoin(page_url, image_url)

    if absolute_url not in image_urls:
        image_urls.append(absolute_url)


def extract_image_urls(page_url):
    print(f"URL: {page_url}")
    print("Sending one request...")

    response = requests.get(
        page_url,
        headers=HEADERS,
        timeout=10,
    )

    print(f"HTTP status: {response.status_code}")

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    image_urls = []

    # Standard <img src="...">
    for image in soup.find_all("img"):
        add_image_url(
            image_urls,
            page_url,
            image.get("src"),
        )

    # Lazy-loaded images
    for image in soup.find_all("img"):
        add_image_url(
            image_urls,
            page_url,
            image.get("data-src"),
        )

    # Open Graph image
    og_image = soup.find(
        "meta",
        attrs={"property": "og:image"},
    )

    if og_image:
        add_image_url(
            image_urls,
            page_url,
            og_image.get("content"),
        )

    # Twitter card image
    twitter_image = soup.find(
        "meta",
        attrs={"name": "twitter:image"},
    )

    if twitter_image:
        add_image_url(
            image_urls,
            page_url,
            twitter_image.get("content"),
        )

    return image_urls


def main():
    if len(sys.argv) != 2:
        print(
            "Usage: python src/extract_image_urls.py "
            "<candidate_url>"
        )
        sys.exit(1)

    page_url = sys.argv[1]

    try:
        image_urls = extract_image_urls(page_url)
    except requests.RequestException as exc:
        print(f"Request failed: {exc}")
        sys.exit(1)

    print()
    print(f"Image URLs found: {len(image_urls)}")

    for index, image_url in enumerate(image_urls, start=1):
        print(f"{index}. {image_url}")


if __name__ == "__main__":
    main()
