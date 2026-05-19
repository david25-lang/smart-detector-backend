from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from routes import api_router
from services.cnn_service import CnnService
from services.yolo_service import YoloService
from settings.config import get_settings
from utils.file_utils import ensure_directories
from utils.logger import setup_logging

settings = get_settings()
setup_logging(settings.LOG_LEVEL)
logger = logging.getLogger("sentry-ai")

ensure_directories(settings.upload_dir_path, settings.output_dir_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models once at startup for reuse.
    logger.info("Loading models")
    app.state.settings = settings
    app.state.yolo_service = YoloService(
        model_path=settings.yolo_model_path,
        conf_threshold=settings.YOLO_CONF_THRESHOLD,
        iou_threshold=settings.YOLO_IOU_THRESHOLD,
        max_det=settings.YOLO_MAX_DETECTIONS,
        img_size=settings.YOLO_IMG_SIZE,
        device=settings.YOLO_DEVICE,
        output_dir=settings.output_dir_path,
        save_output=settings.SAVE_YOLO_OUTPUT,
    )
    app.state.cnn_service = CnnService(
        model_path=settings.cnn_model_path,
        image_size=settings.CNN_IMAGE_SIZE,
        class_names=settings.cnn_class_names_list,
    )
    yield
    logger.info("Shutdown complete")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    docs_url=settings.DOCS_URL,
    redoc_url=settings.REDOC_URL,
    openapi_url=settings.OPENAPI_URL,
    lifespan=lifespan,
)

allow_credentials = False if settings.allow_all_origins else True

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=settings.output_dir_path), name="outputs")


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": {"message": str(exc.detail)}},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": {"message": "Validation error", "details": exc.errors()},
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    logger.exception("Unhandled error")
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": {"message": "Internal server error"}},
    )


@app.get("/", tags=["Health"])
async def root():
    return {"success": True, "message": "Sentry AI backend running"}


@app.get("/health", tags=["Health"])
async def health():
    return {
        "success": True,
        "status": "ok",
        "yolo_loaded": app.state.yolo_service is not None,
        "cnn_loaded": app.state.cnn_service is not None,
    }

app.include_router(api_router, prefix=settings.API_PREFIX)

import os

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
    )