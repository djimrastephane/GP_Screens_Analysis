"""Pixel value normalisation for model input."""

from __future__ import annotations

import numpy as np

# ImageNet channel statistics (RGB order)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_float32(img: np.ndarray) -> np.ndarray:
    """Scale uint8 [0, 255] → float32 [0.0, 1.0]."""
    return img.astype(np.float32) / 255.0


def to_imagenet(img: np.ndarray) -> np.ndarray:
    """Scale and normalise with ImageNet mean/std for pretrained backbone input.

    Input: uint8 RGB HxWx3.
    Output: float32 HxWx3, per-channel zero-mean unit-variance.
    """
    f = to_float32(img)
    return (f - _IMAGENET_MEAN) / _IMAGENET_STD


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert float32 [0, 1] back to uint8 [0, 255] for saving or display."""
    clipped = np.clip(img * 255.0, 0, 255)
    return clipped.astype(np.uint8)


def from_imagenet(img: np.ndarray) -> np.ndarray:
    """Reverse ImageNet normalisation back to uint8 [0, 255]."""
    denorm = img * _IMAGENET_STD + _IMAGENET_MEAN
    return to_uint8(denorm)
