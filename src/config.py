from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT_DIR / "images"
OUTPUT_DIR = ROOT_DIR / "outputs"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

IMAGE_INPUT_DIR = IMAGE_DIR / "input"
VIDEO_INPUT_DIR = IMAGE_DIR / "video"

IMAGE_OUTPUT_DIR = OUTPUT_DIR / "image"
VIDEO_OUTPUT_DIR = OUTPUT_DIR / "output"
WEBCAM_OUTPUT_DIR = OUTPUT_DIR / "webcam"

HAAR_MODEL = CHECKPOINT_DIR / "haarcascade_frontalface_default.xml"
YOLO_MODEL = CHECKPOINT_DIR / "yolov8n.pt"

SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 5
MIN_SIZE = (30, 30)

BOX_COLOR = (0, 255, 0)
BOX_THICKNESS = 2

TEXT_COLOR = (255, 255, 255)
TEXT_SCALE = 0.6
TEXT_THICKNESS = 2

WEBCAM_INDEX = 0

SUPPORTED_IMAGES = [".jpg", ".jpeg", ".png"]
SUPPORTED_VIDEOS = [".mp4", ".avi", ".mov"]

CONFIDENCE = 0.5