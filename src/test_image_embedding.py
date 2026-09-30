from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "image_embedding"
    / "image_classification_mobilenetv2_2022apr.onnx"
)

IMAGE_PATH = PROJECT_ROOT / "data" / "originals" / "img_001.jpg"

FEATURE_TENSOR = "472"


def main():
    net = cv2.dnn.readNetFromONNX(str(MODEL_PATH))

    # Explicit CPU execution.
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise RuntimeError(f"Could not read image: {IMAGE_PATH}")

    blob = cv2.dnn.blobFromImage(
        image,
        scalefactor=1.0 / 127.5,
        size=(224, 224),
        mean=(127.5, 127.5, 127.5),
        swapRB=True,
        crop=True,
    )

    net.setInput(blob)

    # Extract the tensor immediately before the final classifier.
    features = net.forward(FEATURE_TENSOR)

    vector = features.flatten().astype(np.float32)

    print("Model:", MODEL_PATH.name)
    print("Feature tensor:", FEATURE_TENSOR)
    print("Raw shape:", features.shape)
    print("Feature dimension:", len(vector))

    print("\nFirst 10 raw values:")
    print(vector[:10])

    # L2 normalization.
    norm = np.linalg.norm(vector)

    if norm == 0:
        raise RuntimeError("Feature vector has zero norm")

    normalized = vector / norm

    print("\nNormalized dimension:", len(normalized))
    print("L2 norm:", np.linalg.norm(normalized))

    print("\nFirst 10 normalized values:")
    print(normalized[:10])


if __name__ == "__main__":
    main()
