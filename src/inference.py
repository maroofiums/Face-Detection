import time
from pathlib import Path

import cv2

from src.detector import FaceDetector
from src.draw import (
    draw_multiple_faces,
    draw_count,
    draw_fps,
)
from src.utils import (
    load_image,
    save_image,
    show_image,
)


class FaceInference:
    """
    High-level inference pipeline.
    """

    def __init__(self) -> None:
        self.detector = FaceDetector()

    def detect_image(
        self,
        image_path: str | Path,
        save: bool = True,
        show: bool = True,
    ):
        """
        Detect faces in a single image.
        """

        image = load_image(image_path)

        faces, count = self.detector.detect_and_count(image)

        draw_multiple_faces(image, faces)
        draw_count(image, count)

        if save:
            save_image(image, Path(image_path).name)

        if show:
            show_image("Face Detection", image)

        return faces

    def run_webcam(
        self,
        camera_index: int = 0,
    ) -> None:
        """
        Real-time webcam inference.
        """

        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            raise RuntimeError("Unable to open webcam.")

        while True:

            success, frame = cap.read()

            if not success:
                break

            start = time.time()

            faces, count = self.detector.detect_and_count(frame)

            draw_multiple_faces(frame, faces)
            draw_count(frame, count)

            fps = 1 / (time.time() - start)

            draw_fps(frame, fps)

            cv2.imshow(
                "Face Detection",
                frame,
            )

            key = cv2.waitKey(1)

            if key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def run_video(
        self,
        video_path: str | Path,
    ) -> None:
        """
        Run face detection on a video file.
        """

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError("Cannot open video.")

        while True:

            success, frame = cap.read()

            if not success:
                break

            start = time.time()

            faces, count = self.detector.detect_and_count(frame)

            draw_multiple_faces(frame, faces)
            draw_count(frame, count)

            fps = 1 / (time.time() - start)

            draw_fps(frame, fps)

            cv2.imshow(
                "Video Face Detection",
                frame,
            )

            if cv2.waitKey(1) == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()