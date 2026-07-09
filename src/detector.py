"""
Face Detector
-------------
Haar Cascade based face detector.
"""

from pathlib import Path

import cv2
import numpy as np

from src.config import (
    HAAR_MODEL,
    SCALE_FACTOR,
    MIN_NEIGHBORS,
    MIN_SIZE,
)
from src.preprocess import preprocess_for_haar


class FaceDetector:
    """
    Haar Cascade Face Detector.
    """

    def __init__(
        self,
        model_path: str | Path = HAAR_MODEL,
    ) -> None:

        self.model_path = Path(model_path)

        self.detector = cv2.CascadeClassifier(
            str(self.model_path)
        )

        if self.detector.empty():
            raise FileNotFoundError(
                f"Could not load Haar Cascade: {self.model_path}"
            )

    def detect(
        self,
        image: np.ndarray,
    ) -> list[tuple[int, int, int, int]]:
        """
        Detect faces in an image.

        Args:
            image: BGR image.

        Returns:
            List of bounding boxes.
        """

        gray = preprocess_for_haar(image)

        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=SCALE_FACTOR,
            minNeighbors=MIN_NEIGHBORS,
            minSize=MIN_SIZE,
        )

        return [
            tuple(map(int, face))
            for face in faces
        ]

    def detect_and_count(
        self,
        image: np.ndarray,
    ) -> tuple[
        list[tuple[int, int, int, int]],
        int,
    ]:
        """
        Detect faces and return total count.
        """

        faces = self.detect(image)

        return faces, len(faces)