from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
from ultralytics import YOLO


@dataclass
class Detection:
    box_xyxy: Tuple[float, float, float, float]
    score: float
    label: str = "person"


class PersonDetector:
    """Wrap YOLO person detection with configurable backend."""

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        conf: float = 0.5,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.conf = conf
        # ultralytics supports torch/onnx/tensorrt transparently
        self.model = YOLO(model_path)
        if device:
            self.model.to(device)

    def detect(self, image: Image.Image) -> List[Detection]:
        results = self.model(image, conf=self.conf, classes=[0])
        detections: List[Detection] = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                score = float(box.conf[0].cpu())
                detections.append(Detection(box_xyxy=(x1, y1, x2, y2), score=score))
        return detections

    @staticmethod
    def expand_box(
        box_xyxy: Iterable[float],
        expand_ratio: float,
        img_width: int,
        img_height: int,
    ) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = map(float, box_xyxy)
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        dx = width * expand_ratio
        dy = height * expand_ratio

        nx1 = max(0, int(round(x1 - dx)))
        ny1 = max(0, int(round(y1 - dy)))
        nx2 = min(img_width - 1, int(round(x2 + dx)))
        ny2 = min(img_height - 1, int(round(y2 + dy)))

        if nx2 <= nx1:
            nx2 = min(img_width - 1, nx1 + 1)
        if ny2 <= ny1:
            ny2 = min(img_height - 1, ny1 + 1)
        return nx1, ny1, nx2, ny2

    @staticmethod
    def ensure_min_short_side(
        crop: np.ndarray, target_short_side: int
    ) -> Tuple[np.ndarray, float, float]:
        height, width = crop.shape[:2]
        if height == 0 or width == 0:
            return crop, 1.0, 1.0

        short_side = min(height, width)
        if short_side >= target_short_side:
            return crop, 1.0, 1.0

        scale = target_short_side / short_side
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        resized = np.array(
            Image.fromarray(crop).resize((new_width, new_height), Image.BICUBIC)
        )
        scale_x = new_width / width
        scale_y = new_height / height
        return resized, scale_x, scale_y
