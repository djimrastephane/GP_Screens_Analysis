"""ImageMetadata dataclass — matches DATA_CONTRACT image metadata table."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from .quality import QualityReport


@dataclass
class ImageMetadata:
    image_id: str
    source_filename: str
    source_path: str
    file_size_kb: float
    width: int
    height: int
    image_mode: str                  # RGB, RGBA, L, etc.
    image_format: str                # JPEG, PNG, …
    laplacian_variance: float
    mean_brightness: float
    brightness_std: float
    quality_flag: str                # ok / blurry / low_resolution / …
    is_processable: bool
    ingested_at: str                 # ISO-8601 UTC

    # Optional fields — filled later or from user input
    capture_date: str | None = None
    well_name: str | None = None
    completion_zone: str | None = None
    screen_type: str | None = None
    screen_position: str | None = None
    scale_reference_present: bool | None = None
    notes: str | None = None


def extract_metadata(
    path: Path,
    image_id: str,
    pil_img: Image.Image,
    quality: QualityReport,
) -> ImageMetadata:
    """Build an ImageMetadata from already-loaded image and quality report."""
    fmt = pil_img.format or path.suffix.lstrip(".").upper()
    return ImageMetadata(
        image_id=image_id,
        source_filename=path.name,
        source_path=str(path.resolve()),
        file_size_kb=round(os.path.getsize(path) / 1024, 2),
        width=quality.width,
        height=quality.height,
        image_mode=pil_img.mode,
        image_format=fmt,
        laplacian_variance=round(quality.laplacian_variance, 2),
        mean_brightness=round(quality.mean_brightness, 2),
        brightness_std=round(quality.brightness_std, 2),
        quality_flag=quality.quality_flag,
        is_processable=quality.is_processable,
        ingested_at=datetime.now(timezone.utc).isoformat(),
    )
