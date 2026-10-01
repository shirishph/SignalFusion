#!/usr/bin/env python3

import json
import sys
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse
import hashlib

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

def candidate_exists(path, candidate_url):
    """Return True if this candidate URL is already stored."""

    if not path.exists():
        return False

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            if record.get("candidate_url") == candidate_url:
                return True

    return False

def generate_candidate_id(candidate_url):
    """Generate a stable candidate ID from the candidate URL."""

    digest = hashlib.sha256(
        candidate_url.encode("utf-8")
    ).hexdigest()

    return f"C_{digest[:12]}"

def clean_result_url(url):
    """Convert a DuckDuckGo redirect URL into the destination URL."""

    if url.startswith("//"):
        url = "https:" + url

    parsed = urlparse(url)
    query_params = parse_qs(parsed.query)

    if "uddg" in query_params:
        return unquote(query_params["uddg"][0])

    return url


def extract_domain(url):
    """Extract hostname from a URL."""

    parsed = urlparse(url)
    return parsed.netloc


def search_one(query):
    params = {
        "q": query,
    }

    response = requests.get(
        SEARCH_URL,
        params=params,
        headers=HEADERS,
        timeout=10,
    )

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    result = soup.select_one(".result")

    if result is None:
        return None

    title_element = result.select_one(".result__a")
    snippet_element = result.select_one(".result__snippet")

    title = (
        title_element.get_text(" ", strip=True)
        if title_element
        else ""
    )

    snippet = (
        snippet_element.get_text(" ", strip=True)
        if snippet_element
        else ""
    )

    raw_url = ""

    if title_element and title_element.get("href"):
        raw_url = title_element["href"]

    url = clean_result_url(raw_url)

    return {
        "title": title,
        "url": url,
        "snippet": snippet,
    }


def build_candidate(query_id, query_type, query, result):
    candidate_url = result["url"]

    return {
        "query_id": query_id,
        "candidate_id": generate_candidate_id(candidate_url),
        "candidate_url": candidate_url,
        "image_url": "",
        "source": "duckduckgo",
        "discovery_method": query_type,
        "query": query,
        "domain": extract_domain(candidate_url),
        "title": result["title"],
        "snippet": result["snippet"],
    }


def main():
    if len(sys.argv) != 4:
        print(
            "Usage: python src/discover_one.py "
            "<query_id> <query_type> <query>"
        )
        sys.exit(1)

    query_id = sys.argv[1]
    query_type = sys.argv[2]
    query = sys.argv[3]

    print(f"Query ID: {query_id}")
    print(f"Query type: {query_type}")
    print(f"Query: {query}")
    print("Sending one request...")

    try:
        result = search_one(query)
    except requests.RequestException as exc:
        print(f"Request failed: {exc}")
        sys.exit(1)

    if result is None:
        print("No result found.")
        sys.exit(0)

    candidate = build_candidate(
        query_id,
        query_type,
        query,
        result,
    )

    output_path = Path(
        "data/candidates/candidates.jsonl"
    )

    if candidate_exists(output_path, candidate["candidate_url"]):
        print()
        print("Candidate already exists. Not adding duplicate.")
        print(json.dumps(candidate, indent=2))
        sys.exit(0)

    with output_path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(candidate, ensure_ascii=False)
            + "\n"
        )

    print()
    print("Candidate saved:")
    print(json.dumps(candidate, indent=2))

if __name__ == "__main__":
    main()
