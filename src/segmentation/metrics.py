"""Mask area metrics and composite mask construction."""

from __future__ import annotations

import numpy as np


def mask_area_px(mask: np.ndarray) -> int:
    """Count foreground pixels (value > 0)."""
    return int((mask > 0).sum())


def mask_area_pct(mask: np.ndarray) -> float:
    """Foreground area as percentage of total image area (0–100)."""
    h, w = mask.shape[:2]
    return mask_area_px(mask) / (h * w) * 100.0


def bbox_area_px(x1: float, y1: float, x2: float, y2: float) -> int:
    return max(0, int(x2 - x1)) * max(0, int(y2 - y1))


def fill_ratio(mask: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> float:
    """Fraction of the bounding box covered by the mask (0–1)."""
    b_area = bbox_area_px(x1, y1, x2, y2)
    if b_area == 0:
        return 0.0
    # Count mask pixels inside bbox only
    ix1, iy1 = int(x1), int(y1)
    ix2, iy2 = int(x2), int(y2)
    h, w = mask.shape[:2]
    ix1, iy1 = max(0, ix1), max(0, iy1)
    ix2, iy2 = min(w, ix2), min(h, iy2)
    crop = mask[iy1:iy2, ix1:ix2]
    return mask_area_px(crop) / b_area


def union_mask(masks: list[np.ndarray]) -> np.ndarray:
    """Pixel-wise union of a list of binary masks. Returns uint8 0/255."""
    if not masks:
        raise ValueError("masks list is empty")
    result = np.zeros_like(masks[0], dtype=np.uint8)
    for m in masks:
        result = np.maximum(result, m)
    return result
