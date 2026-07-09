from pathlib import Path
import shutil
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.schemas import (
    BoundingBox,
    DetectionResponse,
)
from src.draw import draw_multiple_faces
from src.detector import FaceDetector
from src.utils import (
    is_image_file,
    load_image,
    save_image
)

app = FastAPI(
    title="Face Detection API",
    description="Detect human faces using OpenCV Haar Cascade.",
    version="1.0.0",
)

detector = FaceDetector()


@app.get("/")
def root():
    """
    Health check endpoint.
    """

    return {
        "message": "Face Detection API is running."
    }


@app.post(
    "/detect",
    response_model=DetectionResponse,
)
async def detect_faces(
    file: UploadFile = File(...),
):
    """
    Upload an image and detect faces.
    """

    if not is_image_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format.",
        )

    suffix = Path(file.filename).suffix

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temp:

        shutil.copyfileobj(
            file.file,
            temp,
        )

        temp_path = Path(temp.name)

    try:

        image = load_image(temp_path)

        faces = detector.detect(image)

        # Draw bounding boxes
        draw_multiple_faces(image, faces)

        # Save processed image
        output_path = save_image(
            image,
            f"detected_{file.filename}"
        )

        results = [
            BoundingBox(
                x=x,
                y=y,
                width=w,
                height=h,
            )
            for x, y, w, h in faces
        ]

        return {
            "total_faces": len(results),
            "faces": results,
            "saved_image": str(output_path)
        }

    finally:

        if temp_path.exists():
            temp_path.unlink()