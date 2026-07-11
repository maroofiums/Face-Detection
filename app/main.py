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

from src.config import (
    IMAGE_INPUT_DIR,
    VIDEO_INPUT_DIR,
    IMAGE_OUTPUT_DIR,
    VIDEO_OUTPUT_DIR,
    WEBCAM_OUTPUT_DIR,
)

from src.inference import FaceInference
from src.utils import (
    is_image_file,
    is_video_file,
)

app = FastAPI(
    title="Face Detection API",
    description="Face Detection using OpenCV Haar Cascade",
    version="1.0.0",
)

pipeline = FaceInference()


@app.get(
    "/",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
)
async def home():
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
    if not is_image_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format."
        )

    image_path = IMAGE_INPUT_DIR / file.filename

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    faces = pipeline.detect_image(
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
    if not is_video_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format."
        )

    video_path = VIDEO_INPUT_DIR / file.filename

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    pipeline.run_video(video_path)

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

    pipeline.run_webcam()

    return WebcamResponse(
        status="success",
        message="Webcam detection completed.",
        output_directory=str(
            WEBCAM_OUTPUT_DIR
        ),
    )