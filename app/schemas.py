from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """
    Represents a detected face.
    """

    x: int = Field(..., description="Top-left x coordinate")
    y: int = Field(..., description="Top-left y coordinate")
    width: int = Field(..., description="Bounding box width")
    height: int = Field(..., description="Bounding box height")


class DetectionResponse(BaseModel):
    """
    API response for face detection.
    """

    total_faces: int = Field(
        ...,
        description="Number of detected faces",
    )

    faces: list[BoundingBox]