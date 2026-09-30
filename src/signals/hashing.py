import hashlib
from pathlib import Path

import imagehash
from PIL import Image


def sha256_file(path: Path) -> str:
    """Return SHA-256 hash of the raw file bytes."""
    sha256 = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def perceptual_hashes(path: Path) -> dict:
    """Return pHash and dHash for an image."""
    with Image.open(path) as image:
        return {
            "phash": str(imagehash.phash(image)),
            "dhash": str(imagehash.dhash(image)),
        }
