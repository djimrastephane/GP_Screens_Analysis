"""Classical CV contour detector for GP screen damage.

Two complementary strategies:
  1. Dark-blob detection  — finds erosion holes / breaches that appear as
     dark regions against the bright wire-wrap mesh.
  2. Colour-anomaly detection — finds rust/corrosion patches using HSV
     hue-saturation masking (orange-brown range).

Both strategies use morphological operations to suppress the wire-wrap
repeating pattern before finding candidate damage contours.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .base import BaseDetector, Detection
from .nms import nms

# Class IDs used by this detector
_CLS_DARK_BLOB = 0
_CLS_COLOR_ANOMALY = 1
_CLASS_NAMES = {_CLS_DARK_BLOB: "dark_blob", _CLS_COLOR_ANOMALY: "color_anomaly"}

# HSV hue ranges for rust/corrosion: orange-red (10–30°) and red wrap-around
_RUST_LOWER1 = np.array([5, 80, 50], dtype=np.uint8)
_RUST_UPPER1 = np.array([30, 255, 220], dtype=np.uint8)
_RUST_LOWER2 = np.array([160, 80, 50], dtype=np.uint8)
_RUST_UPPER2 = np.array([180, 255, 220], dtype=np.uint8)


@dataclass
class ContourConfig:
    # Morphological close kernel: suppresses wire-gap noise before thresholding
    close_kernel_size: int = 21
    # Morphological open kernel: removes tiny blobs after threshold
    open_kernel_size: int = 5
    # Adaptive threshold block size (must be odd)
    adaptive_block_size: int = 51
    # Adaptive threshold constant (subtracted from mean)
    adaptive_c: int = 8
    # Minimum blob area as fraction of image area
    min_area_frac: float = 0.002    # 0.2% of image
    # Maximum blob area as fraction of image area
    max_area_frac: float = 0.20     # 20% of image
    # Aspect ratio limits (width/height)
    min_aspect: float = 0.15
    max_aspect: float = 6.5
    # NMS IoU threshold
    nms_iou: float = 0.4
    # Whether to run colour-anomaly strategy
    run_color_anomaly: bool = True


def _confidence_from_area(area_px: int, img_area: int) -> float:
    """Heuristic: blobs covering ~5% of image get confidence=1.0; scales linearly."""
    return min(1.0, (area_px / img_area) / 0.05)


def _detect_dark_blobs(
    img: np.ndarray, cfg: ContourConfig
) -> list[tuple[int, int, int, int, int]]:
    """Return list of (x1, y1, x2, y2, area_px) for dark-blob candidates."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Blur to smooth wire texture before morphological ops
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Morphological close: merges nearby wire gaps into the bright background
    k_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (cfg.close_kernel_size, cfg.close_kernel_size)
    )
    closed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, k_close)

    # Adaptive threshold on the closed image: dark damage stands out
    thresh = cv2.adaptiveThreshold(
        closed, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        cfg.adaptive_block_size,
        cfg.adaptive_c,
    )

    # Morphological open: removes residual small wire-gap noise
    k_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (cfg.open_kernel_size, cfg.open_kernel_size)
    )
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k_open)

    contours, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    h, w = img.shape[:2]
    img_area = h * w
    min_area = cfg.min_area_frac * img_area
    max_area = cfg.max_area_frac * img_area

    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area <= area <= max_area):
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bh == 0:
            continue
        ar = bw / bh
        if not (cfg.min_aspect <= ar <= cfg.max_aspect):
            continue
        results.append((x, y, x + bw, y + bh, int(area)))

    return results


def _detect_color_anomalies(
    img: np.ndarray, cfg: ContourConfig
) -> list[tuple[int, int, int, int, int]]:
    """Return (x1, y1, x2, y2, area_px) for rust/corrosion colour candidates."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    mask1 = cv2.inRange(hsv, _RUST_LOWER1, _RUST_UPPER1)
    mask2 = cv2.inRange(hsv, _RUST_LOWER2, _RUST_UPPER2)
    mask = cv2.bitwise_or(mask1, mask2)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = img.shape[:2]
    img_area = h * w
    min_area = cfg.min_area_frac * img_area
    max_area = cfg.max_area_frac * img_area

    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area <= area <= max_area):
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        results.append((x, y, x + bw, y + bh, int(area)))

    return results


class ContourDetector(BaseDetector):
    """Classical CV detector using morphological cleanup + contour analysis."""

    def __init__(self, config: ContourConfig | None = None):
        self._cfg = config or ContourConfig()

    @property
    def name(self) -> str:
        return "ContourDetector"

    @property
    def version(self) -> str | None:
        return "1.0"

    def detect(self, img: np.ndarray) -> list[Detection]:
        h, w = img.shape[:2]
        img_area = h * w

        detections: list[Detection] = []

        for x1, y1, x2, y2, area in _detect_dark_blobs(img, self._cfg):
            conf = _confidence_from_area(area, img_area)
            detections.append(Detection(
                x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                confidence=conf,
                class_id=_CLS_DARK_BLOB,
                class_name=_CLASS_NAMES[_CLS_DARK_BLOB],
                area_px=area,
                area_pct=round(area / img_area * 100, 4),
                source="contour_dark",
            ))

        if self._cfg.run_color_anomaly:
            for x1, y1, x2, y2, area in _detect_color_anomalies(img, self._cfg):
                conf = _confidence_from_area(area, img_area)
                detections.append(Detection(
                    x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                    confidence=conf,
                    class_id=_CLS_COLOR_ANOMALY,
                    class_name=_CLASS_NAMES[_CLS_COLOR_ANOMALY],
                    area_px=area,
                    area_pct=round(area / img_area * 100, 4),
                    source="contour_color",
                ))

        return nms(detections, iou_threshold=self._cfg.nms_iou)
