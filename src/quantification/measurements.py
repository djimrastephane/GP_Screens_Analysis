"""Geometric and physical measurements from mask areas."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml


def equivalent_diameter_px(area_px: int | float) -> float:
    """Diameter (px) of a circle with the same area as the defect."""
    return math.sqrt(4.0 * area_px / math.pi) if area_px > 0 else 0.0


def pixels_to_mm(px: float, pixels_per_mm: float) -> float:
    """Convert a pixel measurement to millimetres."""
    return px / pixels_per_mm if pixels_per_mm > 0 else 0.0


def screen_region_area_px(
    target_w: int, target_h: int,
    pad_top: int, pad_bottom: int,
    pad_left: int, pad_right: int,
) -> int:
    """Pixel area of the actual image content after removing letterbox padding."""
    content_w = max(0, target_w - pad_left - pad_right)
    content_h = max(0, target_h - pad_top - pad_bottom)
    return content_w * content_h


def corrected_erosion_pct(defect_area_px: int, screen_area_px: int) -> float:
    """Erosion % relative to screen content area (excludes letterbox padding).

    More accurate than dividing by the full padded image area because the
    padding contributes neutral grey pixels that are never part of the screen.
    """
    if screen_area_px <= 0:
        return 0.0
    return min(100.0, defect_area_px / screen_area_px * 100.0)


def defect_density(n_defects: int, screen_area_px: int) -> float:
    """Number of defects per 10,000 screen pixels (normalised count)."""
    if screen_area_px <= 0:
        return 0.0
    return n_defects / screen_area_px * 10_000.0


def load_scale(config_path: Path | None) -> float | None:
    """Return pixels_per_mm from config, or None if not calibrated."""
    if config_path is None or not config_path.exists():
        return None
    with config_path.open() as f:
        raw = yaml.safe_load(f)
    val = raw.get("scale", {}).get("pixels_per_mm")
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None
