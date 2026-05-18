from fastapi import APIRouter, File, Request, UploadFile

from schemas.predictions import CnnClassifyResponse, CnnResult
from utils.image_utils import validate_image_upload

router = APIRouter(tags=["Classification"])


@router.post("/classify", response_model=CnnClassifyResponse)
async def classify(request: Request, image: UploadFile = File(...)):
    settings = request.app.state.settings
    image_bytes = await validate_image_upload(image, settings)

    cnn_service = request.app.state.cnn_service
    predicted_class, confidence = cnn_service.predict(image_bytes)

    result = CnnResult(predicted_class=predicted_class, confidence=confidence)
    return CnnClassifyResponse(success=True, data=result)
