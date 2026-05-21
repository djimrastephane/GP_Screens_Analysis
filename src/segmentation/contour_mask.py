"""ContourMaskSegmenter: re-runs threshold logic within each detection bbox.

For dark_blob detections: Otsu threshold (inverse) identifies the darkest
connected region inside the box. For color_anomaly detections: HSV rust mask.

This approach is fast, requires no model, and directly mirrors the contour
detector's assumptions about what constitutes a defect pixel.
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import BaseSegmenter

# Minimum margin (px) added around bbox before cropping for context
_MARGIN = 12

# HSV rust ranges (same as contour.py)
_RUST_LOWER1 = np.array([5, 80, 50], dtype=np.uint8)
_RUST_UPPER1 = np.array([30, 255, 220], dtype=np.uint8)
_RUST_LOWER2 = np.array([160, 80, 50], dtype=np.uint8)
_RUST_UPPER2 = np.array([180, 255, 220], dtype=np.uint8)


def _dark_blob_mask(crop: np.ndarray) -> np.ndarray:
    """Otsu inverse threshold: returns binary mask of dark pixels in crop."""
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Keep only the largest connected component — avoids wire-gap noise
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh)
    if num_labels <= 1:
        return thresh

    # Largest component excluding background (label 0)
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(thresh)
    out[labels == largest_label] = 255
    return out


def _color_anomaly_mask(crop: np.ndarray) -> np.ndarray:
    """HSV rust mask within the crop."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    m1 = cv2.inRange(hsv, _RUST_LOWER1, _RUST_UPPER1)
    m2 = cv2.inRange(hsv, _RUST_LOWER2, _RUST_UPPER2)
    mask = cv2.bitwise_or(m1, m2)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)


class ContourMaskSegmenter(BaseSegmenter):
    """Fast segmenter that re-applies threshold logic within each detected bbox."""

    @property
    def name(self) -> str:
        return "ContourMaskSegmenter"

    @property
    def version(self) -> str | None:
        return "1.0"

    def segment_one(
        self,
        img: np.ndarray,
        x1: float, y1: float, x2: float, y2: float,
        class_name: str,
    ) -> np.ndarray:
        h, w = img.shape[:2]
        full_mask = np.zeros((h, w), dtype=np.uint8)

        # Expand bbox with margin for context
        cx1 = max(0, int(x1) - _MARGIN)
        cy1 = max(0, int(y1) - _MARGIN)
        cx2 = min(w, int(x2) + _MARGIN)
        cy2 = min(h, int(y2) + _MARGIN)

        crop = img[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return full_mask

        crop_mask = (
            _color_anomaly_mask(crop)
            if class_name == "color_anomaly"
            else _dark_blob_mask(crop)
        )

        # Clip mask back to original bbox (remove margin contribution)
        inner_y1 = int(y1) - cy1
        inner_x1 = int(x1) - cx1
        inner_y2 = inner_y1 + (int(y2) - int(y1))
        inner_x2 = inner_x1 + (int(x2) - int(x1))

        inner_y1 = max(0, inner_y1)
        inner_x1 = max(0, inner_x1)
        inner_y2 = min(crop_mask.shape[0], inner_y2)
        inner_x2 = min(crop_mask.shape[1], inner_x2)

        roi = crop_mask[inner_y1:inner_y2, inner_x1:inner_x2]

        dest_y1 = int(y1)
        dest_x1 = int(x1)
        dest_y2 = dest_y1 + roi.shape[0]
        dest_x2 = dest_x1 + roi.shape[1]

        dest_y2 = min(h, dest_y2)
        dest_x2 = min(w, dest_x2)
        roi = roi[: dest_y2 - dest_y1, : dest_x2 - dest_x1]

        full_mask[dest_y1:dest_y2, dest_x1:dest_x2] = roi
        return full_mask
