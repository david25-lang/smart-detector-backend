from typing import Optional

from fastapi import APIRouter, File, Query, Request, UploadFile

from schemas.predictions import (
    CnnResult,
    CompareResponse,
    CompareResult,
    YoloResult,
)
from utils.file_utils import relative_to_base
from utils.image_utils import save_upload_bytes, validate_image_upload

router = APIRouter(tags=["Comparison"])


@router.post("/compare", response_model=CompareResponse)
async def compare(
    request: Request,
    image: UploadFile = File(...),
    conf_threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
):
    settings = request.app.state.settings
    image_bytes = await validate_image_upload(image, settings)
    upload_path = save_upload_bytes(
        image_bytes, settings.upload_dir_path, image.filename, image.content_type
    )

    yolo_service = request.app.state.yolo_service
    detections, output_path = yolo_service.predict(upload_path, conf_threshold)

    cnn_service = request.app.state.cnn_service
    predicted_class, confidence = cnn_service.predict(image_bytes)

    output_rel = None
    output_url = None
    if output_path:
        output_rel = relative_to_base(output_path, settings.BASE_DIR)
        output_url = f"/{output_rel}"

    threshold = (
        conf_threshold if conf_threshold is not None else settings.YOLO_CONF_THRESHOLD
    )

    yolo_result = YoloResult(
        detections=detections,
        confidence_threshold=threshold,
        output_image_path=output_rel,
        output_image_url=output_url,
        num_detections=len(detections),
    )

    cnn_result = CnnResult(predicted_class=predicted_class, confidence=confidence)

    yolo_classes = [item.class_name for item in detections]
    agreement = predicted_class in yolo_classes if yolo_classes else False

    result = CompareResult(
        yolo=yolo_result,
        cnn=cnn_result,
        agreement=agreement,
        yolo_detected_classes=yolo_classes,
    )

    return CompareResponse(success=True, data=result)
