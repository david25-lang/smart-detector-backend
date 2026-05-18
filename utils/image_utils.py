from io import BytesIO
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import HTTPException, UploadFile
from PIL import Image

from settings.config import Settings

_ALLOWED_EXTENSION_MAP = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


def _get_extension(filename: Optional[str], content_type: Optional[str]) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        return ".jpg" if suffix == ".jpeg" else suffix
    if content_type in _ALLOWED_EXTENSION_MAP:
        return _ALLOWED_EXTENSION_MAP[content_type]
    return ".jpg"


async def validate_image_upload(upload_file: UploadFile, settings: Settings) -> bytes:
    if upload_file.content_type not in settings.allowed_image_types_list:
        raise HTTPException(status_code=400, detail="Unsupported image type")

    data = await upload_file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail="Image too large")

    try:
        with Image.open(BytesIO(data)) as img:
            img.verify()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    return data


def save_upload_bytes(
    data: bytes,
    upload_dir: Path,
    original_filename: Optional[str],
    content_type: Optional[str],
) -> Path:
    extension = _get_extension(original_filename, content_type)
    filename = f"{uuid4().hex}{extension}"
    path = upload_dir / filename
    path.write_bytes(data)
    return path
