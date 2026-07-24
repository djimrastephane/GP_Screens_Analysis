"""YOLOv8 detector wrapper.

Uses ultralytics YOLOv8. If custom weights are not present the detector sets
ready=False and returns no detections — it never silently falls back to COCO
weights, which would return irrelevant class predictions.

Usage once custom weights are trained:
    detector = YOLOv8Detector(weights_path="models/gp_screen_v1.pt")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger

from .base import BaseDetector, Detection
from .nms import nms


class YOLOv8Detector(BaseDetector):
    """Wrapper around ultralytics YOLO for GP screen failure detection."""

    def __init__(
        self,
        weights_path: str | Path | None = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        class_names: dict[int, str] | None = None,
    ):
        self._weights_path = Path(weights_path) if weights_path else None
        self._conf = conf_threshold
        self._iou = iou_threshold
        self._device = device
        self._class_names = class_names or {0: "dark_blob", 1: "color_anomaly"}
        self._model = None
        self._model_version: str | None = None
        self._ready = False

        self._try_load()

    def _try_load(self) -> None:
        if self._weights_path is None:
            logger.warning(
                "YOLOv8Detector: no weights_path provided. "
                "Train a custom model and pass weights_path= to enable detection."
            )
            return

        if not self._weights_path.exists():
            logger.warning(
                f"YOLOv8Detector: weights not found at {self._weights_path}. "
                "Detector will return no detections until weights are available."
            )
            return

        try:
            from ultralytics import YOLO  # type: ignore

            # SECURITY: YOLO() deserializes .pt files via torch.load/pickle,
            # which can execute arbitrary code for a maliciously crafted
            # weights file. weights_path must only ever point to a file from
            # a trusted source (local training output or a vetted internal
            # registry) — never to a user-uploaded or externally-fetched path.
            self._model = YOLO(str(self._weights_path))
            self._model_version = f"YOLOv8/{self._weights_path.name}"
            self._ready = True
            logger.info(f"YOLOv8Detector loaded: {self._weights_path}")
        except ImportError:
            logger.warning(
                "ultralytics is not installed. "
                "Run: pip install ultralytics"
            )

    @property
    def name(self) -> str:
        return "YOLOv8Detector"

    @property
    def version(self) -> str | None:
        return self._model_version

    @property
    def ready(self) -> bool:
        return self._ready

    def detect(self, img: np.ndarray) -> list[Detection]:
        if not self._ready or self._model is None:
            return []

        h, w = img.shape[:2]
        img_area = h * w

        results = self._model.predict(
            img,
            conf=self._conf,
            iou=self._iou,
            device=self._device,
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                area = int((x2 - x1) * (y2 - y1))
                detections.append(Detection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=conf,
                    class_id=cls_id,
                    class_name=self._class_names.get(cls_id, f"class_{cls_id}"),
                    area_px=area,
                    area_pct=round(area / img_area * 100, 4),
                    source="yolo",
                ))

        return nms(detections, iou_threshold=self._iou)
