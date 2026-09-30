from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


class FaceSignalExtractor:
    def __init__(
        self,
        detector_model: Path,
        recognizer_model: Path,
    ):
        self.detector = cv2.FaceDetectorYN.create(
            str(detector_model),
            "",
            (320, 320),
            0.9,
            0.3,
            5000,
        )

        self.recognizer = cv2.FaceRecognizerSF.create(
            str(recognizer_model),
            "",
        )

    def extract(self, image_path: Path) -> dict:
        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        height, width = image.shape[:2]

        self.detector.setInputSize((width, height))

        _, faces = self.detector.detect(image)

        if faces is None:
            return {"faces": []}

        results = []

        for index, face in enumerate(faces):
            # YuNet returns:
            # [x, y, width, height, landmarks..., confidence]
            x, y, w, h = face[:4]
            confidence = float(face[-1])

            aligned_face = self.recognizer.alignCrop(image, face)

            embedding = self.recognizer.feature(aligned_face)

            # Convert from NumPy array to JSON-serializable list.
            embedding = embedding.flatten().astype(float).tolist()

            results.append(
                {
                    "index": index,
                    "bbox": [
                        int(x),
                        int(y),
                        int(w),
                        int(h),
                    ],
                    "confidence": confidence,
                    "embedding": {
                        "model": "opencv-sface-2021dec",
                        "dimension": len(embedding),
                        "vector": embedding,
                    },
                }
            )

        return {"faces": results}
