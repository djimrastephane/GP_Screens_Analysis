"""Image quality assessment: blur, exposure, resolution, glare."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml


@dataclass
class QualityReport:
    laplacian_variance: float
    mean_brightness: float
    brightness_std: float
    width: int
    height: int
    flags: list[str] = field(default_factory=list)

    @property
    def quality_flag(self) -> str:
        return ", ".join(self.flags) if self.flags else "ok"

    @property
    def is_processable(self) -> bool:
        """False only when the image cannot support any meaningful inference."""
        return "low_resolution" not in self.flags or len(self.flags) < 3


def _load_config(config_path: Path | None) -> dict:
    if config_path and config_path.exists():
        with config_path.open() as f:
            return yaml.safe_load(f)
    return {}


def assess_quality(
    img_array: np.ndarray,
    config_path: Path | None = None,
) -> QualityReport:
    """Run all quality checks and return a QualityReport.

    Args:
        img_array: uint8 RGB numpy array.
        config_path: optional path to severity_config.yaml.
    """
    cfg = _load_config(config_path)
    blur_thresh = cfg.get("blur_threshold", 100.0)
    min_w = cfg.get("min_resolution", {}).get("width", 640)
    min_h = cfg.get("min_resolution", {}).get("height", 480)

    h, w = img_array.shape[:2]
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_bright = float(gray.mean())
    bright_std = float(gray.std())

    flags: list[str] = []

    if lap_var < blur_thresh:
        flags.append("blurry")

    if w < min_w or h < min_h:
        flags.append("low_resolution")

    # Glare: saturated pixels (>250 in all channels) above 5% of image
    saturated = np.all(img_array > 250, axis=2)
    if saturated.mean() > 0.05:
        flags.append("glare")

    if mean_bright < 40:
        flags.append("underexposed")

    return QualityReport(
        laplacian_variance=lap_var,
        mean_brightness=mean_bright,
        brightness_std=bright_std,
        width=w,
        height=h,
        flags=flags,
    )
