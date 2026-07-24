"""Classical CV contour detector for GP screen damage.

Two complementary strategies:
  1. Dark-blob detection  — finds erosion holes / breaches that appear as
     dark regions against the bright wire-wrap mesh.
  2. Colour-anomaly detection — finds rust/corrosion patches using HSV
     hue-saturation masking (orange-brown range).

Both strategies use morphological operations to suppress the wire-wrap
repeating pattern before finding candidate damage contours.

Confidence is a weighted blend of three independent classical-CV signals —
saturating blob size, local contrast (dark blobs) or colour saturation
(rust patches), and shape solidity (contour area / convex-hull area) — not
a calibrated probability. It's a relative ranking signal for triage, not a
statistically validated likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .base import BaseDetector, Detection
from .nms import nms_global

# Class IDs used by this detector
_CLS_DARK_BLOB = 0
_CLS_COLOR_ANOMALY = 1
_CLASS_NAMES = {_CLS_DARK_BLOB: "dark_blob", _CLS_COLOR_ANOMALY: "color_anomaly"}

# HSV hue ranges for rust/corrosion: orange-red (5–35°) and red wrap-around.
# Saturation min lowered to 50 to catch faded/weathered corrosion patches.
_RUST_LOWER1 = np.array([5, 50, 40], dtype=np.uint8)
_RUST_UPPER1 = np.array([35, 255, 230], dtype=np.uint8)
_RUST_LOWER2 = np.array([155, 50, 40], dtype=np.uint8)
_RUST_UPPER2 = np.array([180, 255, 230], dtype=np.uint8)


@dataclass
class ContourConfig:
    # Morphological close kernel: suppresses wire-gap noise before thresholding
    close_kernel_size: int = 21
    # Morphological open kernel: removes tiny blobs after threshold
    open_kernel_size: int = 5
    # Adaptive threshold block size (must be odd)
    adaptive_block_size: int = 51
    # Adaptive threshold constant (subtracted from mean) — lower = more sensitive
    adaptive_c: int = 6
    # Minimum blob area as fraction of image area — 0.1% catches small pits
    min_area_frac: float = 0.001
    # Maximum blob area as fraction of image area — 40% allows large damage regions
    max_area_frac: float = 0.40
    # Aspect ratio limits (width/height)
    min_aspect: float = 0.12
    max_aspect: float = 7.0
    # Within-class NMS IoU threshold
    nms_iou: float = 0.35
    # Cross-class IoU threshold for global NMS
    cross_iou: float = 0.50
    # Containment threshold: suppress inner if this fraction is inside a larger box
    containment_threshold: float = 0.80
    # Whether to run colour-anomaly strategy
    run_color_anomaly: bool = True


def _area_score(area_px: int, img_area: int) -> float:
    """Saturating size signal: blobs covering ~5% of image area score 1.0."""
    return min(1.0, (area_px / img_area) / 0.05)


def _solidity(cnt: np.ndarray, area: float) -> float:
    """Contour area / convex-hull area.

    Real damage blobs tend to be compact; ragged noise fragments that still
    pass the area/aspect filters have a much lower ratio, so this catches
    false positives that pure area can't distinguish.
    """
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    return area / hull_area if hull_area > 0 else 0.0


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
    # Background reference for contrast scoring: median over the whole frame
    # is robust to the blob itself (which is a small minority of pixels).
    bg_median = float(np.median(gray))

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

        blob_mask = np.zeros((bh, bw), dtype=np.uint8)
        cv2.drawContours(blob_mask, [cnt - [x, y]], -1, 255, thickness=cv2.FILLED)
        blob_mean = cv2.mean(gray[y:y + bh, x:x + bw], mask=blob_mask)[0]
        contrast_score = min(1.0, max(0.0, (bg_median - blob_mean) / bg_median)) if bg_median > 0 else 0.0

        confidence = (
            0.4 * _area_score(area, img_area)
            + 0.35 * contrast_score
            + 0.25 * _solidity(cnt, area)
        )
        results.append((x, y, x + bw, y + bh, int(area), min(1.0, max(0.0, confidence))))

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
    saturation = hsv[:, :, 1]

    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area <= area <= max_area):
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)

        blob_mask = np.zeros((bh, bw), dtype=np.uint8)
        cv2.drawContours(blob_mask, [cnt - [x, y]], -1, 255, thickness=cv2.FILLED)
        mean_sat = cv2.mean(saturation[y:y + bh, x:x + bw], mask=blob_mask)[0]
        # Weakly-saturated pixels near the HSV lower bound (50) are a marginal
        # match to the rust range; strongly saturated patches are a confident one.
        saturation_score = min(1.0, max(0.0, (mean_sat - 50) / (255 - 50)))

        confidence = (
            0.4 * _area_score(area, img_area)
            + 0.3 * _solidity(cnt, area)
            + 0.3 * saturation_score
        )
        results.append((x, y, x + bw, y + bh, int(area), min(1.0, max(0.0, confidence))))

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

        for x1, y1, x2, y2, area, conf in _detect_dark_blobs(img, self._cfg):
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
            for x1, y1, x2, y2, area, conf in _detect_color_anomalies(img, self._cfg):
                detections.append(Detection(
                    x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                    confidence=conf,
                    class_id=_CLS_COLOR_ANOMALY,
                    class_name=_CLASS_NAMES[_CLS_COLOR_ANOMALY],
                    area_px=area,
                    area_pct=round(area / img_area * 100, 4),
                    source="contour_color",
                ))

        return nms_global(
            detections,
            iou_threshold=self._cfg.nms_iou,
            cross_iou_threshold=self._cfg.cross_iou,
            containment_threshold=self._cfg.containment_threshold,
        )
