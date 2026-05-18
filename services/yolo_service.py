import logging
from pathlib import Path
from typing import List, Optional, Tuple
from uuid import uuid4

import cv2
from ultralytics import YOLO

from schemas.predictions import BoundingBox, Detection

logger = logging.getLogger(__name__)


class YoloService:
    def __init__(
        self,
        model_path: Path,
        conf_threshold: float,
        iou_threshold: float,
        max_det: int,
        img_size: int,
        device: str,
        output_dir: Path,
        save_output: bool,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO model not found at {model_path}")

        self.model = YOLO(str(model_path))
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        self.img_size = img_size
        self.device = device
        self.output_dir = output_dir
        self.save_output = save_output

    def predict(
        self, image_path: Path, conf_threshold: Optional[float] = None
    ) -> Tuple[List[Detection], Optional[Path]]:
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold

        results = self.model.predict(
            source=str(image_path),
            conf=conf,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )

        result = results[0]
        detections: List[Detection] = []
        names = result.names

        for box in result.boxes:
            score = float(box.conf.item())
            if score < conf:
                continue

            class_id = int(box.cls.item())
            if isinstance(names, dict):
                class_name = names.get(class_id, str(class_id))
            else:
                class_name = names[class_id] if class_id < len(names) else str(class_id)

            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=score,
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                )
            )

        output_path = None
        if self.save_output:
            # Save annotated image to the outputs directory.
            annotated = result.plot()
            output_name = f"{image_path.stem}_yolo_{uuid4().hex[:8]}.jpg"
            output_path = self.output_dir / output_name
            cv2.imwrite(str(output_path), annotated)

        return detections, output_path
