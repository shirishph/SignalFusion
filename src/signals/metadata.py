from pathlib import Path

from PIL import Image


def extract_metadata(path: Path) -> dict:
    """Extract basic image metadata."""

    stat = path.stat()

    with Image.open(path) as image:
        exif = image.getexif()

        return {
            "filename": path.name,
            "format": image.format,
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "file_size_bytes": stat.st_size,
            "has_exif": bool(exif),
        }
