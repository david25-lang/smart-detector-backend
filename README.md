# smart-detector-backend

FastAPI backend for pothole and crack detection (YOLO) plus classification (CNN).

## Requirements

- Python 3.9+

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure

Create a `.env` file in this folder. Common keys:

```bash
APP_NAME="Sentry AI - Pothole and Crack Intelligence"
APP_VERSION="1.0.0"
LOG_LEVEL=INFO
API_PREFIX=/api/v1
ALLOWED_ORIGINS=*
MAX_UPLOAD_SIZE_MB=10

YOLO_MODEL_PATH=models/yolo/best.pt
YOLO_CONF_THRESHOLD=0.35
YOLO_IOU_THRESHOLD=0.5
YOLO_MAX_DETECTIONS=100
YOLO_IMG_SIZE=640
YOLO_DEVICE=cpu
SAVE_YOLO_OUTPUT=true

CNN_MODEL_PATH=models/cnn/pothole_crack_cnn_model.keras
CNN_IMAGE_SIZE=224
CNN_CLASS_NAMES=pothole,crack
```

## Run

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## API

- Base URL: http://127.0.0.1:8000
- Health: /health
- Docs: /docs
- Endpoints:
	- POST /api/v1/detect
	- POST /api/v1/classify
	- POST /api/v1/compare

## Models

Place weights in `models/`:

- models/yolo/best.pt
- models/cnn/pothole_crack_cnn_model.keras

These files are intentionally ignored by git.
