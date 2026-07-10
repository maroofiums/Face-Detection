from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    """
    Response model for the root endpoint.
    """

    status: str
    message: str


class DetectionResponse(BaseModel):
    """
    Response model for image/video detection.
    """

    filename: str
    faces_detected: int
    output_path: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "filename": "person.jpg",
                "faces_detected": 3,
                "output_path": "outputs/image/person.jpg",
            }
        }
    )


class WebcamResponse(BaseModel):
    """
    Response model for webcam detection.
    """

    status: str
    message: str
    output_directory: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "message": "Webcam detection completed.",
                "output_directory": "outputs/webcam/",
            }
        }
    )


class ErrorResponse(BaseModel):
    """
    Standard error response.
    """

    detail: str