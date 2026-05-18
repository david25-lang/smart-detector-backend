from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str = "Sentry AI - Pothole and Crack Intelligence"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    API_PREFIX: str = "/api/v1"
    DOCS_URL: str = "/docs"
    REDOC_URL: str = "/redoc"
    OPENAPI_URL: str = "/openapi.json"

    ALLOWED_ORIGINS: str = "*"

    UPLOAD_DIR: str = "uploads"
    OUTPUT_DIR: str = "outputs"
    MAX_UPLOAD_SIZE_MB: int = 10

    YOLO_MODEL_PATH: str = "models/yolo/best.pt"
    YOLO_CONF_THRESHOLD: float = 0.35
    YOLO_IOU_THRESHOLD: float = 0.5
    YOLO_MAX_DETECTIONS: int = 100
    YOLO_IMG_SIZE: int = 640
    YOLO_DEVICE: str = "cpu"
    SAVE_YOLO_OUTPUT: bool = True

    CNN_MODEL_PATH: str = "models/cnn/cnn_model.h5"
    CNN_IMAGE_SIZE: int = 224
    CNN_CLASS_NAMES: str = "pothole,crack"

    ALLOWED_IMAGE_TYPES: str = "image/jpeg,image/png,image/webp"

    BASE_DIR: Path = Path(__file__).resolve().parents[1]

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug_mode(cls, value: Any) -> bool:
        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on", "debug", "dev", "development"}:
                return True
            if normalized in {"0", "false", "no", "off", "release", "prod", "production"}:
                return False

        return value

    @property
    def upload_dir_path(self) -> Path:
        return self.BASE_DIR / self.UPLOAD_DIR

    @property
    def output_dir_path(self) -> Path:
        return self.BASE_DIR / self.OUTPUT_DIR

    @property
    def yolo_model_path(self) -> Path:
        return self.BASE_DIR / self.YOLO_MODEL_PATH

    @property
    def cnn_model_path(self) -> Path:
        return self.BASE_DIR / self.CNN_MODEL_PATH

    @property
    def allowed_origins_list(self) -> List[str]:
        if self.ALLOWED_ORIGINS.strip() == "*":
            return ["*"]
        return [item.strip() for item in self.ALLOWED_ORIGINS.split(",") if item.strip()]

    @property
    def allow_all_origins(self) -> bool:
        return self.ALLOWED_ORIGINS.strip() == "*"

    @property
    def allowed_image_types_list(self) -> List[str]:
        return [item.strip() for item in self.ALLOWED_IMAGE_TYPES.split(",") if item.strip()]

    @property
    def cnn_class_names_list(self) -> List[str]:
        return [item.strip() for item in self.CNN_CLASS_NAMES.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
