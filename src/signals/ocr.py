import pytesseract
from PIL import Image
from pathlib import Path


def extract_ocr(path: Path) -> dict:
    """Extract text from an image using local Tesseract OCR."""

    with Image.open(path) as image:
        text = pytesseract.image_to_string(image)

    text = text.strip()

    return {
        "text": text,
        "has_text": bool(text),
    }
