from pathlib import Path
import shutil

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    status,
)

from app.schemas import (
    DetectionResponse,
    HealthResponse,
    WebcamResponse,
)

from src.inference import FaceInference
from src.utils import (
    ensure_directory,
    is_image_file,
    is_video_file,
)

from src.config import (
    IMAGE_INPUT_DIR,
    VIDEO_INPUT_DIR,
    IMAGE_OUTPUT_DIR,
    VIDEO_OUTPUT_DIR,
    WEBCAM_OUTPUT_DIR
)


for directory in (
    IMAGE_INPUT_DIR,
    VIDEO_INPUT_DIR,
    IMAGE_OUTPUT_DIR,
    VIDEO_OUTPUT_DIR,
    WEBCAM_OUTPUT_DIR,
):
    ensure_directory(directory)


app = FastAPI(
    title="Face Detection API",
    description="Face Detection using OpenCV Haar Cascade",
    version="1.0.0",
)

detector = FaceInference()


@app.get(
    "/",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
)
async def home():
    """
    Health check endpoint.
    """

    return HealthResponse(
        status="success",
        message="Face Detection API is running."
    )


@app.post(
    "/detect/image",
    response_model=DetectionResponse,
)
async def detect_image(
    file: UploadFile = File(...)
):
    """
    Detect faces in an uploaded image.
    """

    if not is_image_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format."
        )

    image_path = IMAGE_INPUT_DIR / file.filename

    with image_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    faces = detector.detect_image(
        image_path=image_path,
        save=True,
        show=False,
    )

    return DetectionResponse(
        filename=file.filename,
        faces_detected=len(faces),
        output_path=str(
            IMAGE_OUTPUT_DIR / file.filename
        ),
    )


@app.post(
    "/detect/video",
    response_model=DetectionResponse,
)
async def detect_video(
    file: UploadFile = File(...)
):
    """
    Detect faces in uploaded video.
    """

    if not is_video_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format."
        )

    video_path = VIDEO_INPUT_DIR / file.filename

    with video_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    detector.run_video(video_path)

    return DetectionResponse(
        filename=file.filename,
        faces_detected=0,
        output_path=str(
            VIDEO_OUTPUT_DIR / file.filename
        ),
    )


@app.post(
    "/detect/webcam",
    response_model=WebcamResponse,
)
async def detect_webcam():
    """
    Start webcam face detection.
    """

    detector.run_webcam()

    return WebcamResponse(
        status="success",
        message="Webcam detection completed.",
        output_directory=str(
            WEBCAM_OUTPUT_DIR
        ),
    )