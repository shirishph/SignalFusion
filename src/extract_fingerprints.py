import json
import sys
from pathlib import Path

from signals.hashing import sha256_file, perceptual_hashes
from signals.metadata import extract_metadata
from signals.ocr import extract_ocr
from signals.face import FaceSignalExtractor
from signals.image_embedding import ImageEmbeddingExtractor

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FACE_DETECTOR = (
    PROJECT_ROOT
    / "models"
    / "face"
    / "face_detection_yunet_2023mar.onnx"
)

FACE_RECOGNIZER = (
    PROJECT_ROOT
    / "models"
    / "face"
    / "face_recognition_sface_2021dec.onnx"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

IMAGE_EMBEDDING_MODEL = (
    PROJECT_ROOT
    / "models"
    / "image_embedding"
    / "image_classification_mobilenetv2_2022apr.onnx"
)


def find_images(input_dir: Path):
    """Return all supported images recursively."""
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def extract_fingerprint(
    image_path: Path,
    face_extractor: FaceSignalExtractor,
    image_embedding_extractor: ImageEmbeddingExtractor,
) -> dict:

    record = {
        "image_id": image_path.stem,
        "source": {
            "path": str(image_path),
            "filename": image_path.name,
        },
        "hashes": {},
        "metadata": {},
        "ocr": {},
        "faces": [],
        "image_embedding": {},
        "extraction": {
            "status": "success",
            "errors": {},
        },
    }

    try:
        record["hashes"]["sha256"] = sha256_file(image_path)
        record["hashes"].update(perceptual_hashes(image_path))
    except Exception as e:
        record["extraction"]["errors"]["hashes"] = str(e)

    try:
        record["metadata"] = extract_metadata(image_path)
    except Exception as e:
        record["extraction"]["errors"]["metadata"] = str(e)

    try:
        record["ocr"] = extract_ocr(image_path)
    except Exception as e:
        record["extraction"]["errors"]["ocr"] = str(e)

    try:
        record["faces"] = face_extractor.extract(image_path)["faces"]
    except Exception as e:
        record["extraction"]["errors"]["faces"] = str(e)

    if record["extraction"]["errors"]:
        record["extraction"]["status"] = "partial"

    try:
        record["image_embedding"] = (
            image_embedding_extractor.extract(image_path)
        )
    except Exception as e:
        record["extraction"]["errors"][
            "image_embedding"
        ] = str(e)



    return record


def main():

    if len(sys.argv) != 3:
        print(
            "Usage: python src/extract_fingerprints.py "
            "<input_dir> <output_file>"
        )
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    images = find_images(input_dir)

    print(f"Found {len(images)} images")

    face_extractor = FaceSignalExtractor(
        detector_model=FACE_DETECTOR,
        recognizer_model=FACE_RECOGNIZER,
    )

    image_embedding_extractor = ImageEmbeddingExtractor(
        model_path=IMAGE_EMBEDDING_MODEL,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    successful = 0
    partial = 0

    with output_file.open("w", encoding="utf-8") as f:

        for index, image_path in enumerate(images, start=1):

            print(
                f"[{index}/{len(images)}] "
                f"Processing {image_path.name}"
            )

            record = extract_fingerprint(
                image_path,
                face_extractor,
                image_embedding_extractor,
            )

            f.write(
                json.dumps(record, ensure_ascii=False)
                + "\n"
            )

            if record["extraction"]["status"] == "success":
                successful += 1
            else:
                partial += 1

    print()
    print("Extraction complete")
    print(f"Images processed: {len(images)}")
    print(f"Successful:       {successful}")
    print(f"Partial:          {partial}")
    print(f"Output:           {output_file}")


if __name__ == "__main__":
    main()
