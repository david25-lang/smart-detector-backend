from typing import List, Optional

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBox


class YoloResult(BaseModel):
    detections: List[Detection]
    confidence_threshold: float = Field(..., ge=0.0, le=1.0)
    output_image_path: Optional[str]
    output_image_url: Optional[str]
    num_detections: int


class CnnResult(BaseModel):
    predicted_class: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class CompareResult(BaseModel):
    yolo: YoloResult
    cnn: CnnResult
    agreement: bool
    yolo_detected_classes: List[str]


class YoloDetectResponse(BaseModel):
    success: bool
    data: YoloResult


class CnnClassifyResponse(BaseModel):
    success: bool
    data: CnnResult


class CompareResponse(BaseModel):
    success: bool
    data: CompareResult
