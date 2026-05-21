"""Image discovery, loading, and file-level validation."""

from __future__ import annotations

import re
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}


def discover_images(directory: str | Path) -> list[Path]:
    """Return sorted list of supported image paths in directory (non-recursive)."""
    d = Path(directory)
    if not d.is_dir():
        raise NotADirectoryError(f"Not a directory: {d}")
    return sorted(
        p for p in d.iterdir()
        if p.is_file() and p.suffix.lower() in ALLOWED_SUFFIXES
    )


def validate_file(path: Path) -> tuple[bool, str]:
    """Check file is non-empty and has a supported extension.

    Returns (is_valid, reason).
    """
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        return False, f"unsupported extension: {path.suffix}"
    if path.stat().st_size == 0:
        return False, "zero-byte file"
    return True, "ok"


def load_image(path: Path) -> tuple[Image.Image, np.ndarray]:
    """Load image as PIL Image and uint8 RGB numpy array.

    RGBA images are composited onto white before conversion.
    """
    img = Image.open(path)
    img.load()  # force decode to catch corrupt files early

    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    return img, np.array(img, dtype=np.uint8)


def generate_image_id(path: Path) -> str:
    """Stable slug from filename: lowercase, spaces → underscores, no extension."""
    stem = path.stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    # Append short content hash to handle collisions from renamed files
    digest = hashlib.md5(path.read_bytes()).hexdigest()[:8]
    return f"{stem}_{digest}"
