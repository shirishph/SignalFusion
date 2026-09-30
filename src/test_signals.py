import sys
from pathlib import Path

from signals.hashing import sha256_file, perceptual_hashes
from signals.metadata import extract_metadata
from signals.ocr import extract_ocr
from signals.face import FaceSignalExtractor

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

def main():
    if len(sys.argv) != 2:
        print("Usage: python src/test_signals.py <image>")
        sys.exit(1)

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        print(f"File not found: {image_path}")
        sys.exit(1)

    print("\nSHA-256:")
    print(sha256_file(image_path))

    print("\nPerceptual hashes:")
    print(perceptual_hashes(image_path))

    print("\nMetadata:")
    print(extract_metadata(image_path))

    print("\nOCR:")
    print(extract_ocr(image_path))

    face_extractor = FaceSignalExtractor(
        detector_model=FACE_DETECTOR,
        recognizer_model=FACE_RECOGNIZER,
    )

    print("\nFaces:")
    face_result = face_extractor.extract(image_path)

    print(f"Number of faces: {len(face_result['faces'])}")

    for face in face_result["faces"]:
        print(
            f"  Face {face['index']}: "
            f"bbox={face['bbox']}, "
            f"confidence={face['confidence']:.4f}, "
            f"embedding_dimension={len(face['embedding'])}"
        )


if __name__ == "__main__":
    main()
