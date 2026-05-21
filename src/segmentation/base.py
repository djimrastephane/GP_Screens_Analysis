"""Shared dataclasses and abstract base for all segmenters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SegmentationMask:
    """Pixel-precise defect mask for one detection, in full padded-image space."""
    detection_index: int
    class_id: int
    class_name: str
    detection_confidence: float
    source: str                  # "contour_mask" | "grabcut" | "sam"
    mask: np.ndarray             # HxW uint8, 0 or 255, full padded image size
    defect_area_px: int
    defect_area_pct: float       # fraction of full image area (0–100)
    bbox_area_px: int
    fill_ratio: float            # defect_area / bbox_area — how dense the defect is

    def to_dict(self) -> dict:
        return {
            "detection_index": self.detection_index,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "detection_confidence": round(self.detection_confidence, 4),
            "source": self.source,
            "defect_area_px": self.defect_area_px,
            "defect_area_pct": round(self.defect_area_pct, 4),
            "bbox_area_px": self.bbox_area_px,
            "fill_ratio": round(self.fill_ratio, 4),
        }


@dataclass
class SegmentationResult:
    """All masks produced for one image in one segmenter run."""
    image_id: str
    source_filename: str
    segmenter_name: str
    model_version: str | None
    masks: list[SegmentationMask]
    image_width: int
    image_height: int
    run_timestamp: str
    composite_defect_area_px: int = 0    # union of all masks
    composite_defect_area_pct: float = 0.0

    @property
    def count(self) -> int:
        return len(self.masks)


class BaseSegmenter(ABC):

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def version(self) -> str | None: ...

    @abstractmethod
    def segment_one(
        self,
        img: np.ndarray,
        x1: float, y1: float, x2: float, y2: float,
        class_name: str,
    ) -> np.ndarray:
        """Return a full-image binary mask (HxW uint8 0/255) for one detection bbox."""
        ...

    @property
    def ready(self) -> bool:
        return True
