import cv2
import numpy as np


def resize_image(
    image: np.ndarray,
    width: int | None = None,
    height: int | None = None,
) -> np.ndarray:
    """
    Resize image while preserving aspect ratio.

    Args:
        image: Input image.
        width: Desired width.
        height: Desired height.

    Returns:
        Resized image.
    """

    if width is None and height is None:
        return image

    h, w = image.shape[:2]

    if width is not None:
        ratio = width / w
        new_size = (width, int(h * ratio))
    else:
        ratio = height / h
        new_size = (int(w * ratio), height)

    return cv2.resize(
        image,
        new_size,
        interpolation=cv2.INTER_AREA,
    )


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to grayscale.
    """

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )


def gaussian_blur(
    image: np.ndarray,
    kernel_size: tuple[int, int] = (5, 5),
) -> np.ndarray:
    """
    Apply Gaussian Blur to remove image noise.
    """

    return cv2.GaussianBlur(
        image,
        kernel_size,
        0,
    )


def equalize_histogram(
    image: np.ndarray,
) -> np.ndarray:
    """
    Improve image contrast using histogram equalization.
    """

    return cv2.equalizeHist(image)


def normalize(
    image: np.ndarray,
) -> np.ndarray:
    """
    Normalize pixel values to range [0,1].
    """

    return image.astype(np.float32) / 255.0


def preprocess_for_haar(
    image: np.ndarray,
) -> np.ndarray:
    """
    Complete preprocessing pipeline for Haar Cascade.
    """

    gray = to_grayscale(image)

    gray = gaussian_blur(gray)

    gray = equalize_histogram(gray)

    return gray