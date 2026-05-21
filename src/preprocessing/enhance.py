"""Conditional image enhancement: CLAHE and non-local means denoising."""

from __future__ import annotations

import cv2
import numpy as np


def apply_clahe(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE to the L channel of the LAB colour space.

    Works on RGB uint8 input, returns RGB uint8.
    Improves local contrast without blowing out highlights.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)


def apply_denoise(
    img: np.ndarray,
    h: float = 10.0,
    template_window: int = 7,
    search_window: int = 21,
) -> np.ndarray:
    """Apply fast non-local means denoising (colour).

    Works on RGB uint8 input, returns RGB uint8.
    Only applied when the blurry flag is set — it sharpens apparent texture
    by removing noise while preserving edges.
    """
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    denoised_bgr = cv2.fastNlMeansDenoisingColored(
        bgr,
        None,
        h=h,
        hColor=h,
        templateWindowSize=template_window,
        searchWindowSize=search_window,
    )
    return cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)


def select_enhancements(quality_flags: list[str]) -> list[str]:
    """Return ordered list of enhancement names to apply given quality flags.

    Order matters: denoise before CLAHE so noise reduction doesn't fight
    contrast amplification.
    """
    enhancements: list[str] = []
    if "blurry" in quality_flags:
        enhancements.append("denoise")
    if "underexposed" in quality_flags:
        enhancements.append("clahe")
    return enhancements
