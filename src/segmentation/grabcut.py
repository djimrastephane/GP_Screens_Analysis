"""GrabCutSegmenter: box-initialised GrabCut for precise pixel segmentation.

More accurate than threshold-based methods for irregular shapes. Requires the
bbox to have a margin of at least 8px of background context — very tight crops
(< 30px wide/tall) fall back to ContourMaskSegmenter.
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import BaseSegmenter
from .contour_mask import ContourMaskSegmenter

_FALLBACK = ContourMaskSegmenter()

# Margin added around the detection box before GrabCut
_MARGIN = 20
# Minimum bbox dimension (px) to attempt GrabCut; smaller → fallback
_MIN_DIM = 30
# GrabCut iterations
_ITER = 5


class GrabCutSegmenter(BaseSegmenter):
    """GrabCut segmenter initialised from bounding boxes."""

    def __init__(self, iterations: int = _ITER, margin: int = _MARGIN):
        self._iterations = iterations
        self._margin = margin

    @property
    def name(self) -> str:
        return "GrabCutSegmenter"

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

        box_w = int(x2 - x1)
        box_h = int(y2 - y1)

        # Fall back for tiny boxes — GrabCut needs context
        if box_w < _MIN_DIM or box_h < _MIN_DIM:
            return _FALLBACK.segment_one(img, x1, y1, x2, y2, class_name)

        # Crop with margin
        cx1 = max(0, int(x1) - self._margin)
        cy1 = max(0, int(y1) - self._margin)
        cx2 = min(w, int(x2) + self._margin)
        cy2 = min(h, int(y2) + self._margin)

        crop = img[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return full_mask

        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

        # GrabCut rect is relative to the crop
        rect_x = int(x1) - cx1
        rect_y = int(y1) - cy1
        rect_w = min(box_w, crop.shape[1] - rect_x)
        rect_h = min(box_h, crop.shape[0] - rect_y)

        if rect_w <= 0 or rect_h <= 0:
            return full_mask

        gc_mask = np.zeros(crop_bgr.shape[:2], dtype=np.uint8)
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)

        try:
            cv2.grabCut(
                crop_bgr, gc_mask, (rect_x, rect_y, rect_w, rect_h),
                bgd_model, fgd_model,
                self._iterations, cv2.GC_INIT_WITH_RECT,
            )
        except cv2.error:
            # GrabCut can fail on pathological crops (all-same-colour etc.)
            return _FALLBACK.segment_one(img, x1, y1, x2, y2, class_name)

        # Pixels marked probable-foreground or definite-foreground
        fg_mask = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
            np.uint8(255), np.uint8(0),
        )

        # Place crop mask back into full image
        dest_y1, dest_y2 = cy1, cy2
        dest_x1, dest_x2 = cx1, cx2
        full_mask[dest_y1:dest_y2, dest_x1:dest_x2] = fg_mask
        return full_mask
