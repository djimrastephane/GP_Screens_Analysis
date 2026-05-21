"""Shared dataclasses and abstract base for all detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detection:
    """Single bounding-box detection in padded image coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str
    area_px: int
    area_pct: float        # fraction of image area, 0–100
    source: str            # "contour_dark" | "contour_color" | "yolo"

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "x1": round(self.x1, 2),
            "y1": round(self.y1, 2),
            "x2": round(self.x2, 2),
            "y2": round(self.y2, 2),
            "confidence": round(self.confidence, 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "area_px": self.area_px,
            "area_pct": round(self.area_pct, 4),
            "source": self.source,
        }


@dataclass
class DetectionResult:
    """All detections for one image from one detector run."""
    image_id: str
    source_filename: str
    detector_name: str
    model_version: str | None
    detections: list[Detection]
    image_width: int
    image_height: int
    run_timestamp: str
    notes: str | None = None

    @property
    def count(self) -> int:
        return len(self.detections)

    @property
    def max_confidence(self) -> float:
        return max((d.confidence for d in self.detections), default=0.0)


class BaseDetector(ABC):
    """All detectors implement this interface."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def version(self) -> str | None: ...

    @abstractmethod
    def detect(self, img: np.ndarray) -> list[Detection]:
        """Run detection on a uint8 RGB array, return detections in image coordinates."""
        ...

    @property
    def ready(self) -> bool:
        return True
