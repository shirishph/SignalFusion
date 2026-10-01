#!/usr/bin/env python3

import sys
from urllib.parse import parse_qs, unquote, urlparse

import requests
from bs4 import BeautifulSoup


SEARCH_URL = "https://html.duckduckgo.com/html/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/153.0.0.0 Safari/537.36"
    )
}


def clean_result_url(url):
    """Convert a DuckDuckGo redirect URL into the destination URL."""

    if url.startswith("//"):
        url = "https:" + url

    parsed = urlparse(url)

    query_params = parse_qs(parsed.query)

    if "uddg" in query_params:
        return unquote(query_params["uddg"][0])

    return url


def search_one(query):
    params = {
        "q": query,
    }

    print(f"Query: {query}")
    print("Sending one request...")

    response = requests.get(
        SEARCH_URL,
        params=params,
        headers=HEADERS,
        timeout=10,
    )

    print(f"HTTP status: {response.status_code}")

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    result = soup.select_one(".result")

    if result is None:
        print("No search result found.")
        return None

    title_element = result.select_one(".result__a")
    snippet_element = result.select_one(".result__snippet")

    title = title_element.get_text(" ", strip=True) if title_element else ""

    snippet = (
        snippet_element.get_text(" ", strip=True)
        if snippet_element
        else ""
    )

    raw_url = ""

    if title_element and title_element.get("href"):
        raw_url = title_element["href"]

    url = clean_result_url(raw_url)

    print()
    print("First result:")
    print(f"  Title:   {title}")
    print(f"  URL:     {url}")
    print(f"  Snippet: {snippet}")

    return {
        "title": title,
        "url": url,
        "snippet": snippet,
    }


def main():
    if len(sys.argv) != 2:
        print(
            "Usage: python src/test_web_search.py "
            '"search query"'
        )
        sys.exit(1)

    query = sys.argv[1]

    try:
        search_one(query)
    except requests.RequestException as exc:
        print(f"Request failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
