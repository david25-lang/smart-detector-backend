import logging
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

logger = logging.getLogger(__name__)


def _patch_dense_from_config() -> None:
    if getattr(Dense, "_sentry_ai_patched", False):
        return

    original_from_config = Dense.from_config

    @classmethod
    def patched_from_config(cls, config):
        config = dict(config)
        config.pop("quantization_config", None)
        return original_from_config(config)

    Dense.from_config = patched_from_config
    Dense._sentry_ai_patched = True


class CnnService:
    def __init__(self, model_path: Path, image_size: int, class_names: List[str]) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"CNN model not found at {model_path}")

        _patch_dense_from_config()
        self.model = load_model(str(model_path), compile=False)
        self.image_size = image_size
        self.class_names = class_names

    def _preprocess(self, image_bytes: bytes) -> np.ndarray:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image = image.resize((self.image_size, self.image_size))
        array = np.asarray(image, dtype=np.float32) / 255.0
        return np.expand_dims(array, axis=0)

    def predict(self, image_bytes: bytes) -> Tuple[str, float]:
        inputs = self._preprocess(image_bytes)
        preds = self.model.predict(inputs, verbose=0)

        if preds.ndim == 2 and preds.shape[1] == 1:
            prob = float(preds[0][0])
            index = 1 if prob >= 0.5 else 0
            class_name = (
                self.class_names[index] if len(self.class_names) > index else str(index)
            )
            confidence = prob if index == 1 else 1.0 - prob
            return class_name, confidence

        probs = preds[0]
        index = int(np.argmax(probs))
        confidence = float(probs[index])
        class_name = (
            self.class_names[index] if len(self.class_names) > index else str(index)
        )
        return class_name, confidence
