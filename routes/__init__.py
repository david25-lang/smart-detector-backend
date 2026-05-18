from fastapi import APIRouter

from routes.classify import router as classify_router
from routes.compare import router as compare_router
from routes.detect import router as detect_router

api_router = APIRouter()
api_router.include_router(detect_router)
api_router.include_router(classify_router)
api_router.include_router(compare_router)
