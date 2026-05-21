"""Feature extraction from a segmentation mask and the underlying image region."""

from __future__ import annotations

import cv2
import numpy as np

from .base import FeatureVector


def extract_features(
    img: np.ndarray,
    mask: np.ndarray,
    detection: dict,
) -> FeatureVector:
    """Compute shape and colour features for one masked region.

    Args:
        img: uint8 RGB full padded image.
        mask: uint8 HxW binary mask (0/255) in full image space.
        detection: detection dict with keys x1, y1, x2, y2, source, area_pct.

    Returns:
        FeatureVector with shape and colour features.
    """
    # --- Shape from contour ---
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Fall back to bbox-derived values if no contour
    if not cnts:
        return _fallback_features(detection)

    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 1:
        return _fallback_features(detection)

    perimeter = cv2.arcLength(cnt, True)
    circularity = (4 * np.pi * area / perimeter ** 2) if perimeter > 0 else 0.0

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    x, y, bw, bh = cv2.boundingRect(cnt)
    aspect_ratio = bw / bh if bh > 0 else 0.0
    bbox_area = bw * bh
    extent = area / bbox_area if bbox_area > 0 else 0.0

    h_img, w_img = mask.shape[:2]
    img_area = h_img * w_img
    area_pct = area / img_area * 100.0

    # --- Colour at masked pixels ---
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    px = hsv[mask > 0]

    if len(px) == 0:
        mean_hue = std_hue = mean_sat = mean_val = 0.0
    else:
        mean_hue = float(px[:, 0].mean())
        std_hue = float(px[:, 0].std())
        mean_sat = float(px[:, 1].mean())
        mean_val = float(px[:, 2].mean())

    return FeatureVector(
        circularity=round(circularity, 4),
        solidity=round(solidity, 4),
        aspect_ratio=round(aspect_ratio, 4),
        extent=round(extent, 4),
        area_pct=round(area_pct, 4),
        mean_hue=round(mean_hue, 2),
        std_hue=round(std_hue, 2),
        mean_saturation=round(mean_sat, 2),
        mean_value=round(mean_val, 2),
        detection_source=detection.get("source", "unknown"),
    )


def _fallback_features(detection: dict) -> FeatureVector:
    """Return a feature vector derived from bbox geometry when no contour exists."""
    x1, y1 = detection.get("x1", 0), detection.get("y1", 0)
    x2, y2 = detection.get("x2", 0), detection.get("y2", 0)
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    return FeatureVector(
        circularity=0.0,
        solidity=0.0,
        aspect_ratio=bw / bh,
        extent=0.5,
        area_pct=detection.get("area_pct", 0.0),
        mean_hue=0.0,
        std_hue=0.0,
        mean_saturation=0.0,
        mean_value=0.0,
        detection_source=detection.get("source", "unknown"),
    )
