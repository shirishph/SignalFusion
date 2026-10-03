import asyncio
import html
import json
import re
from pathlib import Path
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse, unquote

import requests
from PicImageSearch import Network
from PicImageSearch.engines import Yandex

from io import BytesIO
from PIL import Image


IMAGE = "query_image/img_007.jpg"

MATCH_COUNT = 5
SIMILAR_COUNT = 5

CANDIDATES_DIR = Path("candidates")
CANDIDATES_JSON = CANDIDATES_DIR / "candidates.json"


def make_similar_url(yandex_url):
    parsed = urlparse(yandex_url)
    params = parse_qs(parsed.query)
    params["cbir_page"] = ["similar"]

    new_query = urlencode(params, doseq=True)

    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            new_query,
            parsed.fragment,
        )
    )


def extract_similar_items(raw_html):
    decoded = html.unescape(raw_html)

    items = []

    pattern = re.compile(
        r'"img_href"\s*:\s*"([^"]+)"'
        r'.{0,3000}?'
        r'"serpItemSource"\s*:\s*"IMAGESCBIR_SIMILAR"',
        re.DOTALL,
    )

    for match in pattern.finditer(decoded):
        image_url = match.group(1)

        image_url = (
            image_url
            .replace("\\/", "/")
            .replace("\\u002F", "/")
        )

        block = match.group(0)

        pos_match = re.search(
            r'"pos"\s*:\s*(\d+)',
            block,
        )

        alt_match = re.search(
            r'"alt"\s*:\s*"([^"]*)"',
            block,
        )

        items.append(
            {
                "position": int(pos_match.group(1))
                if pos_match
                else None,
                "title": alt_match.group(1)
                if alt_match
                else "",
                "image_url": image_url,
            }
        )

    # Remove duplicate image URLs while preserving order.
    unique = []
    seen = set()

    for item in items:
        url = item["image_url"]

        if url in seen:
            continue

        seen.add(url)
        unique.append(item)

    return unique


def get_filename(image_url, fallback):
    """
    Extract the original filename from the image URL.
    """

    path = urlparse(image_url).path
    filename = Path(unquote(path)).name

    if not filename:
        filename = fallback

    return filename


def make_unique_filename(filename, used_filenames):
    """
    Preserve the original filename.

    If the same filename occurs more than once,
    append _2, _3, etc. rather than overwriting.
    """

    if filename not in used_filenames:
        used_filenames.add(filename)
        return filename

    path = Path(filename)

    stem = path.stem
    suffix = path.suffix

    counter = 2

    while True:
        candidate = f"{stem}_{counter}{suffix}"

        if candidate not in used_filenames:
            used_filenames.add(candidate)
            return candidate

        counter += 1


def download_image(image_url, output_path):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(
        image_url,
        headers=headers,
        timeout=30,
    )

    response.raise_for_status()

    data = response.content

    # ------------------------------------------------------------
    # Validate that the response is actually an image.
    # ------------------------------------------------------------

    try:
        with Image.open(BytesIO(data)) as image:
            image_format = image.format

    except Exception as exc:
        raise ValueError(
            f"Response is not a valid image "
            f"(Content-Type: "
            f"{response.headers.get('Content-Type', '')})"
        ) from exc

    # ------------------------------------------------------------
    # Determine extension from the actual image format
    # when URL has no extension.
    # ------------------------------------------------------------

    if not output_path.suffix:
        extension_map = {
            "JPEG": ".jpg",
            "PNG": ".png",
            "WEBP": ".webp",
            "GIF": ".gif",
            "BMP": ".bmp",
            "TIFF": ".tiff",
        }

        extension = extension_map.get(image_format)

        if extension:
            output_path = output_path.with_name(
                output_path.name + extension
            )

    output_path.write_bytes(data)

    return output_path

async def main():

    CANDIDATES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates = []
    used_filenames = set()

    async with Network() as client:

        engine = Yandex(client=client)

        # ------------------------------------------------------------
        # 1. Reverse-image matches
        # ------------------------------------------------------------

        result = await engine.search(file=IMAGE)

        matches = result.raw[:MATCH_COUNT]

        print("=" * 70)
        print("Yandex matches")
        print("=" * 70)

        for item in matches:

            original = item.origin.get(
                "originalImage",
                {},
            )

            image_url = original.get("url")

            if not image_url:
                continue

            source_url = item.url

            filename = get_filename(
                image_url,
                "candidate.jpg",
            )

            filename = make_unique_filename(
                filename,
                used_filenames,
            )

            candidates.append(
                {
                    "cid": f"c{len(candidates) + 1:04d}",
                    "url": source_url,
                    "image_filename": filename,
                    "_image_url": image_url,
                }
            )

        # ------------------------------------------------------------
        # 2. Visually similar images
        # ------------------------------------------------------------

        similar_url = make_similar_url(result.url)

        print()
        print("Fetching Yandex similar-image results...")

        response = requests.get(
            similar_url,
            headers={
                "User-Agent": "Mozilla/5.0"
            },
            timeout=30,
        )

        response.raise_for_status()

        similar_items = extract_similar_items(
            response.text
        )

        print(
            f"Found {len(similar_items)} "
            "similar-image records"
        )

        for item in similar_items[:SIMILAR_COUNT]:

            image_url = item["image_url"]

            filename = get_filename(
                image_url,
                "candidate.jpg",
            )

            filename = make_unique_filename(
                filename,
                used_filenames,
            )

            candidates.append(
                {
                    "cid": f"c{len(candidates) + 1:04d}",
                    "url": image_url,
                    "image_filename": filename,
                    "_image_url": image_url,
                }
            )

    # ------------------------------------------------------------
    # 3. Download candidates
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("Downloading candidates")
    print("=" * 70)

    final_candidates = []
    index = 1

    for candidate in candidates:

        # cid = candidate["cid"]
        cid = "c000" + str(index)
        source_url = candidate["url"]
        image_url = candidate["_image_url"]
        filename = candidate["image_filename"]

        output_path = CANDIDATES_DIR / filename

        print()
        print(f"{cid}")
        print(f"Image URL : {image_url}")
        print(f"Filename  : {filename}")

        try:

            output_path = download_image(
                image_url,
                output_path,
            )

            # Use the actual filename written to disk.
            filename = output_path.name

            print("Status    : downloaded")
            print(f"Saved as  : {filename}")

            final_candidates.append(
                {
                    "cid": cid,
                    "url": source_url,
                    "image_filename": filename,
                }
            )
            index = index + 1

        except Exception as exc:

            print(f"Status    : FAILED")
            print(f"Error     : {exc}")

    # ------------------------------------------------------------
    # 4. Write candidates.json
    # ------------------------------------------------------------

    CANDIDATES_JSON.write_text(
        json.dumps(
            final_candidates,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(f"Candidates downloaded : {len(final_candidates)}")
    print(f"Directory             : {CANDIDATES_DIR}")
    print(f"JSON                  : {CANDIDATES_JSON}")


if __name__ == "__main__":
    asyncio.run(main())
