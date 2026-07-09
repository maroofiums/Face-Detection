import cv2
import numpy as np

from src.config import (
    BOX_COLOR,
    BOX_THICKNESS,
    TEXT_COLOR,
    TEXT_SCALE,
    TEXT_THICKNESS,
)


def draw_face_box(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    label: str = "Face",
) -> np.ndarray:
    """
    Draw a single face bounding box.

    Args:
        image: Input image.
        box: (x, y, w, h)
        label: Display label.

    Returns:
        Image with bounding box.
    """

    x, y, w, h = box

    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        BOX_COLOR,
        BOX_THICKNESS,
    )

    cv2.putText(
        image,
        label,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        TEXT_SCALE,
        TEXT_COLOR,
        TEXT_THICKNESS,
    )

    return image


def draw_multiple_faces(
    image: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
) -> np.ndarray:
    """
    Draw multiple detected faces.

    Args:
        image: Input image.
        boxes: List of face boxes.

    Returns:
        Image with all boxes.
    """

    for box in boxes:
        draw_face_box(image, box)

    return image


def draw_confidence(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    confidence: float,
) -> np.ndarray:
    """
    Draw confidence score above the box.

    Args:
        image: Input image.
        box: Bounding box.
        confidence: Detection confidence.

    Returns:
        Image with confidence score.
    """

    x, y, _, _ = box

    text = f"{confidence:.2f}"

    cv2.putText(
        image,
        text,
        (x, y - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        TEXT_SCALE,
        TEXT_COLOR,
        TEXT_THICKNESS,
    )

    return image


def draw_fps(
    image: np.ndarray,
    fps: float,
) -> np.ndarray:
    """
    Display FPS on image.

    Args:
        image: Input image.
        fps: Frames per second.

    Returns:
        Image with FPS.
    """

    text = f"FPS: {fps:.2f}"

    cv2.putText(
        image,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    return image


def draw_count(
    image: np.ndarray,
    count: int,
) -> np.ndarray:
    """
    Display number of detected faces.

    Args:
        image: Input image.
        count: Number of faces.

    Returns:
        Image with face count.
    """

    text = f"Faces: {count}"

    cv2.putText(
        image,
        text,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2,
    )

    return image