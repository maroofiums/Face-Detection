from pathlib import Path
import logging

import cv2
import numpy as np

from src.config import (
    OUTPUT_DIR,
    SUPPORTED_IMAGES,
    SUPPORTED_VIDEOS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def ensure_directory(directory: Path) -> None:
    """
    Create a directory if it does not exist.

    Args:
        directory: Path object representing the directory.
    """
    directory.mkdir(parents=True, exist_ok=True)


def load_image(image_path: str | Path) -> np.ndarray:
    """
    Load an image from disk.

    Args:
        image_path: Path to the image.

    Returns:
        Loaded image as NumPy array.

    Raises:
        FileNotFoundError:
            If image does not exist.

        ValueError:
            If OpenCV cannot read the image.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    logger.info("Loaded image: %s", image_path.name)

    return image


def save_image(
    image: np.ndarray,
    filename: str,
) -> Path:
    """
    Save image into outputs folder.

    Args:
        image: Image array.
        filename: Output filename.

    Returns:
        Path of saved image.
    """

    ensure_directory(OUTPUT_DIR)

    output_path = OUTPUT_DIR / filename

    cv2.imwrite(str(output_path), image)

    logger.info("Saved image: %s", output_path)

    return output_path


def is_image_file(path: str | Path) -> bool:
    """
    Check whether file is a supported image.

    Args:
        path: File path.

    Returns:
        True if supported image.
    """
    return Path(path).suffix.lower() in SUPPORTED_IMAGES


def is_video_file(path: str | Path) -> bool:
    """
    Check whether file is a supported video.

    Args:
        path: File path.

    Returns:
        True if supported video.
    """
    return Path(path).suffix.lower() in SUPPORTED_VIDEOS


def image_size(image: np.ndarray) -> tuple[int, int]:
    """
    Return image width and height.

    Args:
        image: Image array.

    Returns:
        (width, height)
    """
    height, width = image.shape[:2]
    return width, height

def show_image(
    window_name: str,
    image: np.ndarray,
) -> None:
    """
    Display image until a key is pressed.

    Args:
        window_name: Window title.
        image: Image array.
    """
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()