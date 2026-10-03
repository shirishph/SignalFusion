import cv2
import imagehash
import json
import numpy as np
import onnxruntime as ort
from pathlib import Path
from PIL import Image
import pytesseract

def preprocess(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    image = image.astype(np.float32) / 255.0

    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    return image

def main():
    print()
    print("pHash")

    candidates_file = Path("candidates/candidates.json")
    candidates_dir = Path("candidates")
    output_dir = Path("data/fingerprints/phash")

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(candidates_file, "r") as f:
        candidates = json.load(f)

    for candidate in candidates:
        cid = candidate["cid"]
        image_filename = candidate["image_filename"]

        image_path = candidates_dir / image_filename
        output_path = output_dir / f"{cid}.json"

        image = Image.open(image_path)
        phash = imagehash.phash(image)

        result = {
            "cid": cid,
            "image_filename": image_filename,
            "phash": str(phash)
        }

        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"Created: {output_path}")

    query_image = Path("query_image/img_007.jpg")
    output_file = Path("data/fingerprints/phash/query_image.json")

    image = Image.open(query_image)
    phash = imagehash.phash(image)

    result = {
        "image_filename": query_image.name,
        "phash": str(phash)
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Created: {output_file}")

    print()
    print("Image Embedding")

    CANDIDATES_FILE = Path("candidates/candidates.json")
    CANDIDATES_DIR = Path("candidates")
    OUTPUT_DIR = Path("data/fingerprints/image_embedding")

    MODEL_PATH = Path(
        "models/image_embedding/image_classification_mobilenetv2_2022apr.onnx"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(CANDIDATES_FILE, "r") as f:
        candidates = json.load(f)

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    for candidate in candidates:
        cid = candidate["cid"]
        image_filename = candidate["image_filename"]

        image_path = CANDIDATES_DIR / image_filename
        output_path = OUTPUT_DIR / f"{cid}.json"

        image = preprocess(image_path)

        embedding = session.run(
            None,
            {input_name: image},
        )[0]

        embedding = embedding.flatten()

        # L2 normalize
        norm = np.linalg.norm(embedding)

        if norm > 0:
            embedding = embedding / norm

        result = {
            "cid": cid,
            "image_filename": image_filename,
            "embedding_dimension": len(embedding),
            "embedding": embedding.tolist(),
        }

        with open(output_path, "w") as f:
            json.dump(result, f)

        print(
            f"Created: {output_path} "
            f"(dimension={len(embedding)})"
        )

    query_image = Path("query_image/img_007.jpg")
    output_path = OUTPUT_DIR / f"query_image.json"

    image = preprocess(query_image)

    embedding = session.run(
        None,
        {input_name: image},
    )[0]

    embedding = embedding.flatten()

    # L2 normalize
    norm = np.linalg.norm(embedding)

    if norm > 0:
        embedding = embedding / norm

    result = {
        "query_image": str(query_image),
        "embedding_dimension": len(embedding),
        "embedding": embedding.tolist(),
    }

    with open(output_path, "w") as f:
        json.dump(result, f)

    print(
        f"Created: {output_path} "
        f"(dimension={len(embedding)})"
    )

    print()
    print("Face Embedding")

    CANDIDATES_FILE = Path("candidates/candidates.json")
    CANDIDATES_DIR = Path("candidates")
    OUTPUT_DIR = Path("data/fingerprints/face_embedding")

    FACE_DETECTION_MODEL = Path(
        "models/face/face_detection_yunet_2023mar.onnx"
    )

    FACE_RECOGNITION_MODEL = Path(
        "models/face/face_recognition_sface_2021dec.onnx"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(CANDIDATES_FILE, "r") as f:
        candidates = json.load(f)

    detector = cv2.FaceDetectorYN.create(
        str(FACE_DETECTION_MODEL),
        "",
        (320, 320),
        0.9,
        0.3,
        5000,
    )

    recognizer = cv2.FaceRecognizerSF.create(
        str(FACE_RECOGNITION_MODEL),
        "",
    )

    for candidate in candidates:
        cid = candidate["cid"]
        image_filename = candidate["image_filename"]

        image_path = CANDIDATES_DIR / image_filename
        output_path = OUTPUT_DIR / f"{cid}.json"

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        height, width = image.shape[:2]
        detector.setInputSize((width, height))

        _, faces = detector.detect(image)

        result = {
            "cid": cid,
            "image_filename": image_filename,
            "faces": []
        }

        if faces is not None:
            for face_index, face in enumerate(faces):
                aligned_face = recognizer.alignCrop(
                    image,
                    face,
                )

                embedding = recognizer.feature(
                    aligned_face
                )

                embedding = embedding.flatten()

                norm = np.linalg.norm(embedding)

                if norm > 0:
                    embedding = embedding / norm

                result["faces"].append({
                    "face_index": face_index,
                    "embedding_dimension": len(embedding),
                    "embedding": embedding.tolist(),
                })

        with open(output_path, "w") as f:
            json.dump(result, f)

        print(
            f"Created: {output_path} "
            f"(faces={len(result['faces'])})"
        )

    query_image = Path("query_image/img_007.jpg")
    output_path = OUTPUT_DIR / f"query_image.json"

    image = cv2.imread(str(query_image))

    if image is None:
        raise ValueError(f"Could not read image: {query_image}")

    height, width = image.shape[:2]
    detector.setInputSize((width, height))

    _, faces = detector.detect(image)

    result = {
        "query_image": str(query_image),
        "faces": []
    }

    if faces is not None:
        for face_index, face in enumerate(faces):
            aligned_face = recognizer.alignCrop(
                image,
                face,
            )

            embedding = recognizer.feature(
                aligned_face
            )

            embedding = embedding.flatten()

            norm = np.linalg.norm(embedding)

            if norm > 0:
                embedding = embedding / norm

            result["faces"].append({
                "face_index": face_index,
                "embedding_dimension": len(embedding),
                "embedding": embedding.tolist(),
            })

    with open(output_path, "w") as f:
        json.dump(result, f)

    print(
        f"Created: {output_path} "
        f"(faces={len(result['faces'])})"
    )

    print()
    print("OCR")

    CANDIDATES_FILE = Path("candidates/candidates.json")
    CANDIDATES_DIR = Path("candidates")
    OUTPUT_DIR = Path("data/fingerprints/ocr")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(CANDIDATES_FILE, "r") as f:
        candidates = json.load(f)

    for candidate in candidates:
        cid = candidate["cid"]
        image_filename = candidate["image_filename"]

        image_path = CANDIDATES_DIR / image_filename
        output_path = OUTPUT_DIR / f"{cid}.json"

        image = Image.open(image_path)

        text = pytesseract.image_to_string(image)

        result = {
            "cid": cid,
            "image_filename": image_filename,
            "text": text.strip()
        }

        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"Created: {output_path}")

    query_image = Path("query_image/img_007.jpg")
    output_path = OUTPUT_DIR / f"query_image.json"

    image = Image.open(query_image)

    text = pytesseract.image_to_string(image)

    result = {
        "query_image": str(query_image),
        "text": text.strip()
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Created: {output_path}")


if __name__ == "__main__":
    main()

