from pathlib import Path

import cv2
import numpy as np


class ImageEmbeddingExtractor:
    """CPU-only MobileNetV2 image feature extractor."""

    def __init__(self, model_path: Path):
        self.model_path = model_path

        self.net = cv2.dnn.readNetFromONNX(
            str(model_path)
        )

        # Explicitly use OpenCV CPU execution.
        self.net.setPreferableBackend(
            cv2.dnn.DNN_BACKEND_OPENCV
        )

        self.net.setPreferableTarget(
            cv2.dnn.DNN_TARGET_CPU
        )

        # Tensor immediately before the final classifier.
        self.feature_tensor = "472"

    def extract(self, image_path: Path) -> dict:
        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(
                f"Could not read image: {image_path}"
            )

        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0 / 127.5,
            size=(224, 224),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
            crop=True,
        )

        self.net.setInput(blob)

        features = self.net.forward(
            self.feature_tensor
        )

        vector = (
            features
            .flatten()
            .astype(np.float32)
        )

        norm = np.linalg.norm(vector)

        if norm == 0:
            raise RuntimeError(
                "Image embedding has zero norm"
            )

        vector = vector / norm

        return {
            "model": "mobilenetv2-2022apr",
            "dimension": int(vector.shape[0]),
            "normalized": True,
            "vector": vector.tolist(),
        }
