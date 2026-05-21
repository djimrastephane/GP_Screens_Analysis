"""Shared dataclasses and abstract base for all classifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class FailureType(str, Enum):
    EROSION_HOLE = "erosion_hole"
    WIRE_WRAP_FAILURE = "wire_wrap_failure"
    SCREEN_COLLAPSE = "screen_collapse"
    CORROSION_PITTING = "corrosion_pitting"
    MECHANICAL_DAMAGE = "mechanical_damage"
    PLUGGING_PARTIAL = "plugging_partial"
    PLUGGING_COMPLETE = "plugging_complete"
    UNKNOWN = "unknown"


@dataclass
class FeatureVector:
    """Shape and colour features extracted from one mask region."""
    # Shape
    circularity: float          # 0–1; 1 = perfect circle
    solidity: float             # filled_area / convex_hull_area
    aspect_ratio: float         # bbox_width / bbox_height
    extent: float               # mask_area / bbox_area
    area_pct: float             # mask area as % of full image

    # Colour (HSV statistics at mask pixels)
    mean_hue: float             # 0–180 (OpenCV)
    std_hue: float
    mean_saturation: float      # 0–255
    mean_value: float           # 0–255 (brightness)

    # Detection context
    detection_source: str       # "contour_dark" | "contour_color" | "yolo"

    def to_dict(self) -> dict:
        return {
            "circularity": round(self.circularity, 4),
            "solidity": round(self.solidity, 4),
            "aspect_ratio": round(self.aspect_ratio, 4),
            "extent": round(self.extent, 4),
            "area_pct": round(self.area_pct, 4),
            "mean_hue": round(self.mean_hue, 2),
            "std_hue": round(self.std_hue, 2),
            "mean_saturation": round(self.mean_saturation, 2),
            "mean_value": round(self.mean_value, 2),
            "detection_source": self.detection_source,
        }


@dataclass
class ClassificationResult:
    """Classification output for one detection."""
    detection_index: int
    class_name: str                   # detection class ("dark_blob" | "color_anomaly")
    failure_type: FailureType
    confidence: float                 # 0–1
    severity: str                     # low | medium | high | critical
    requires_human_review: bool
    reasoning: str                    # brief explanation for audit trail
    features: FeatureVector

    def to_dict(self) -> dict:
        return {
            "detection_index": self.detection_index,
            "class_name": self.class_name,
            "failure_type": self.failure_type.value,
            "confidence": round(self.confidence, 4),
            "severity": self.severity,
            "requires_human_review": self.requires_human_review,
            "reasoning": self.reasoning,
            "features": self.features.to_dict(),
        }


@dataclass
class ImageClassificationResult:
    """All classification results for one image."""
    image_id: str
    source_filename: str
    classifier_name: str
    classifications: list[ClassificationResult]
    run_timestamp: str
    dominant_failure_type: str        # most frequent failure type
    composite_defect_area_pct: float  # from segmentation result
    overall_severity: str             # worst severity across all detections

    @property
    def count(self) -> int:
        return len(self.classifications)

    @property
    def review_count(self) -> int:
        return sum(1 for c in self.classifications if c.requires_human_review)


class BaseClassifier(ABC):

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def classify_one(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        detection: dict,
        composite_defect_area_pct: float,
    ) -> ClassificationResult: ...
