"""Colour palette and visual style constants for annotation overlays."""

from __future__ import annotations

# RGB colour per failure type — chosen for contrast on typical screen imagery
FAILURE_TYPE_RGB: dict[str, tuple[int, int, int]] = {
    "erosion_hole":       (220,  55,  55),   # red
    "corrosion_pitting":  (230, 155,  25),   # amber
    "wire_wrap_failure":  ( 40, 180, 220),   # cyan
    "screen_collapse":    (175,  35, 175),   # purple
    "mechanical_damage":  ( 90, 110, 220),   # blue-purple
    "plugging_partial":   ( 45, 195,  95),   # green
    "plugging_complete":  ( 15,  95,  50),   # dark green
    "unknown":            (150, 150, 150),   # grey
}

# Overall severity — text colour / badge background
SEVERITY_RGB: dict[str, tuple[int, int, int]] = {
    "low":      ( 70, 180,  70),   # green
    "medium":   (230, 160,  20),   # amber
    "high":     (220,  80,  20),   # orange-red
    "critical": (200,  20,  20),   # red
}

# Bounding-box border thickness per severity level (pixels)
SEVERITY_THICKNESS: dict[str, int] = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}

# Mask fill transparency (0 = transparent, 1 = opaque)
MASK_ALPHA = 0.38

# Header bar
HEADER_BG   = (30, 30, 30)     # near-black
HEADER_FG   = (240, 240, 240)  # near-white
HEADER_H    = 36               # pixels

# Legend
LEGEND_BG   = (20, 20, 20)
LEGEND_FG   = (230, 230, 230)
LEGEND_SWATCH_SIZE = 12

# Review-flag badge
REVIEW_BADGE_BG = (220, 140,  20)   # orange
REVIEW_BADGE_FG = (255, 255, 255)


def get_colour(failure_type: str) -> tuple[int, int, int]:
    return FAILURE_TYPE_RGB.get(failure_type, FAILURE_TYPE_RGB["unknown"])


def get_severity_colour(severity: str) -> tuple[int, int, int]:
    return SEVERITY_RGB.get(severity, SEVERITY_RGB["low"])


def rgb_to_bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return (rgb[2], rgb[1], rgb[0])
