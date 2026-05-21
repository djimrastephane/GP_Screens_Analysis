"""Letterbox resize: scale image to target dimensions with neutral-fill padding.

Preserves aspect ratio. Returns the padded array and a PaddingInfo so that
downstream code can map pixel coordinates back to the original image.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PaddingInfo:
    """Geometric transform applied during letterbox resize.

    Use these fields to convert (x, y) coordinates from padded space back to
    original image space:
        x_orig = (x_padded - pad_left) / scale
        y_orig = (y_padded - pad_top) / scale
    """
    original_width: int
    original_height: int
    target_width: int
    target_height: int
    scale: float
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int


def letterbox(
    img: np.ndarray,
    target_width: int = 640,
    target_height: int = 640,
    fill: int = 114,
) -> tuple[np.ndarray, PaddingInfo]:
    """Resize img to (target_width × target_height) with aspect-ratio padding.

    Args:
        img: uint8 HxWxC numpy array (RGB).
        target_width: output width in pixels.
        target_height: output height in pixels.
        fill: fill value for padded pixels (default 114 = YOLOv8 grey).

    Returns:
        Tuple of (padded_array uint8, PaddingInfo).
    """
    h, w = img.shape[:2]
    scale = min(target_width / w, target_height / h)

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    pad_vert = target_height - new_h
    pad_horiz = target_width - new_w
    pad_top = pad_vert // 2
    pad_bottom = pad_vert - pad_top
    pad_left = pad_horiz // 2
    pad_right = pad_horiz - pad_left

    channels = img.shape[2] if img.ndim == 3 else 1
    canvas = np.full((target_height, target_width, channels), fill, dtype=np.uint8)
    canvas[pad_top: pad_top + new_h, pad_left: pad_left + new_w] = resized

    info = PaddingInfo(
        original_width=w,
        original_height=h,
        target_width=target_width,
        target_height=target_height,
        scale=scale,
        pad_top=pad_top,
        pad_bottom=pad_bottom,
        pad_left=pad_left,
        pad_right=pad_right,
    )
    return canvas, info


def unpad_coords(
    x: float, y: float, info: PaddingInfo
) -> tuple[float, float]:
    """Map a (x, y) point from padded space back to original image coordinates."""
    x_orig = (x - info.pad_left) / info.scale
    y_orig = (y - info.pad_top) / info.scale
    return x_orig, y_orig


def unpad_box(
    x1: float, y1: float, x2: float, y2: float, info: PaddingInfo
) -> tuple[float, float, float, float]:
    """Map an xyxy bounding box from padded space back to original image coordinates."""
    x1o, y1o = unpad_coords(x1, y1, info)
    x2o, y2o = unpad_coords(x2, y2, info)
    return x1o, y1o, x2o, y2o
