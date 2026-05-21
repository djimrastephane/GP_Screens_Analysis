"""Unit tests for src/preprocessing modules."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.resize import letterbox, unpad_box, unpad_coords
from src.preprocessing.enhance import apply_clahe, apply_denoise, select_enhancements
from src.preprocessing.normalize import (
    from_imagenet,
    to_float32,
    to_imagenet,
    to_uint8,
)
from src.preprocessing.pipeline import PreprocessingConfig, preprocess_image


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rgb_640() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(30, 200, size=(480, 640, 3), dtype=np.uint8)


@pytest.fixture
def rgb_wide() -> np.ndarray:
    """Wider than tall — tests horizontal letterboxing."""
    rng = np.random.default_rng(7)
    return rng.integers(50, 220, size=(300, 900, 3), dtype=np.uint8)


@pytest.fixture
def rgb_tall() -> np.ndarray:
    """Taller than wide — tests vertical letterboxing."""
    rng = np.random.default_rng(3)
    return rng.integers(50, 220, size=(900, 300, 3), dtype=np.uint8)


@pytest.fixture
def rgb_small() -> np.ndarray:
    return np.full((80, 100, 3), 128, dtype=np.uint8)


@pytest.fixture
def rgb_dark() -> np.ndarray:
    return np.full((480, 640, 3), 20, dtype=np.uint8)


# ---------------------------------------------------------------------------
# resize
# ---------------------------------------------------------------------------

class TestLetterbox:
    def test_output_shape(self, rgb_640):
        out, _ = letterbox(rgb_640, 640, 640)
        assert out.shape == (640, 640, 3)

    def test_square_target(self, rgb_wide):
        out, info = letterbox(rgb_wide, 640, 640)
        assert out.shape == (640, 640, 3)
        # Wide image (900x300): width is limiting → scale fits width → pad top/bottom
        assert info.pad_top > 0 or info.pad_bottom > 0

    def test_tall_image(self, rgb_tall):
        out, info = letterbox(rgb_tall, 640, 640)
        assert out.shape == (640, 640, 3)
        # Tall image (300x900): height is limiting → scale fits height → pad left/right
        assert info.pad_left > 0 or info.pad_right > 0

    def test_fill_value(self, rgb_small):
        out, info = letterbox(rgb_small, 640, 640, fill=114)
        # Top-left corner of padding should be fill value
        assert out[0, 0, 0] == 114

    def test_padding_sums_to_target(self, rgb_wide):
        _, info = letterbox(rgb_wide, 640, 640)
        assert info.pad_top + info.pad_bottom + round(info.original_height * info.scale) <= 640
        assert info.pad_left + info.pad_right + round(info.original_width * info.scale) <= 640

    def test_scale_less_than_one_for_large_input(self):
        large = np.zeros((1296, 1296, 3), dtype=np.uint8)
        _, info = letterbox(large, 640, 640)
        assert info.scale < 1.0

    def test_scale_greater_than_one_for_small_input(self, rgb_small):
        _, info = letterbox(rgb_small, 640, 640)
        assert info.scale > 1.0

    def test_dtype_preserved(self, rgb_640):
        out, _ = letterbox(rgb_640, 640, 640)
        assert out.dtype == np.uint8


class TestUnpadCoords:
    def test_top_left_corner(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        _, info = letterbox(img, 640, 640)
        x_orig, y_orig = unpad_coords(info.pad_left, info.pad_top, info)
        assert abs(x_orig) < 1.0
        assert abs(y_orig) < 1.0

    def test_unpad_box_roundtrip(self, rgb_640):
        _, info = letterbox(rgb_640, 640, 640)
        # Map a box at centre of padded image back to original space
        cx = info.pad_left + (640 - info.pad_left - info.pad_right) / 2
        cy = info.pad_top + (640 - info.pad_top - info.pad_bottom) / 2
        x1o, y1o, x2o, y2o = unpad_box(cx - 10, cy - 10, cx + 10, cy + 10, info)
        assert x2o > x1o
        assert y2o > y1o


# ---------------------------------------------------------------------------
# enhance
# ---------------------------------------------------------------------------

class TestSelectEnhancements:
    def test_no_flags(self):
        assert select_enhancements([]) == []

    def test_blurry_flag(self):
        assert "denoise" in select_enhancements(["blurry"])

    def test_underexposed_flag(self):
        assert "clahe" in select_enhancements(["underexposed"])

    def test_order_denoise_before_clahe(self):
        result = select_enhancements(["underexposed", "blurry"])
        assert result.index("denoise") < result.index("clahe")

    def test_unrelated_flag_ignored(self):
        result = select_enhancements(["low_resolution"])
        assert result == []


class TestApplyClahe:
    def test_output_shape_preserved(self, rgb_640):
        out = apply_clahe(rgb_640)
        assert out.shape == rgb_640.shape

    def test_dtype_uint8(self, rgb_640):
        out = apply_clahe(rgb_640)
        assert out.dtype == np.uint8

    def test_dark_image_brightened(self, rgb_dark):
        out = apply_clahe(rgb_dark)
        assert out.mean() >= rgb_dark.mean()


class TestApplyDenoise:
    def test_output_shape_preserved(self, rgb_640):
        out = apply_denoise(rgb_640)
        assert out.shape == rgb_640.shape

    def test_dtype_uint8(self, rgb_640):
        out = apply_denoise(rgb_640)
        assert out.dtype == np.uint8


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_float32_range(self, rgb_640):
        f = to_float32(rgb_640)
        assert f.dtype == np.float32
        assert f.min() >= 0.0
        assert f.max() <= 1.0

    def test_to_uint8_roundtrip(self, rgb_640):
        f = to_float32(rgb_640)
        back = to_uint8(f)
        assert back.dtype == np.uint8
        # Allow ±1 rounding error
        assert np.max(np.abs(back.astype(int) - rgb_640.astype(int))) <= 1

    def test_imagenet_output_dtype(self, rgb_640):
        out = to_imagenet(rgb_640)
        assert out.dtype == np.float32

    def test_imagenet_not_bounded_0_1(self, rgb_640):
        out = to_imagenet(rgb_640)
        # ImageNet normalization can produce negative values
        assert out.min() < 0 or out.max() > 1

    def test_from_imagenet_roundtrip(self, rgb_640):
        norm = to_imagenet(rgb_640)
        back = from_imagenet(norm)
        assert back.dtype == np.uint8
        assert np.max(np.abs(back.astype(int) - rgb_640.astype(int))) <= 2


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------

class TestPreprocessImage:
    def test_output_shape_float32(self, rgb_640):
        cfg = PreprocessingConfig(target_width=640, target_height=640)
        f, u, info, enh = preprocess_image(rgb_640, [], cfg)
        assert f.shape == (640, 640, 3)
        assert f.dtype == np.float32

    def test_output_shape_uint8(self, rgb_640):
        cfg = PreprocessingConfig()
        _, u, _, _ = preprocess_image(rgb_640, [], cfg)
        assert u.shape == (640, 640, 3)
        assert u.dtype == np.uint8

    def test_no_enhancements_clean_image(self, rgb_640):
        cfg = PreprocessingConfig()
        _, _, _, enh = preprocess_image(rgb_640, [], cfg)
        assert enh == []

    def test_underexposed_triggers_clahe(self, rgb_640):
        cfg = PreprocessingConfig()
        _, _, _, enh = preprocess_image(rgb_640, ["underexposed"], cfg)
        assert "clahe" in enh

    def test_blurry_triggers_denoise(self, rgb_640):
        cfg = PreprocessingConfig()
        _, _, _, enh = preprocess_image(rgb_640, ["blurry"], cfg)
        assert "denoise" in enh

    def test_custom_target_size(self, rgb_640):
        cfg = PreprocessingConfig(target_width=320, target_height=320)
        f, _, _, _ = preprocess_image(rgb_640, [], cfg)
        assert f.shape == (320, 320, 3)

    def test_imagenet_normalisation(self, rgb_640):
        cfg = PreprocessingConfig(normalisation="imagenet")
        f, _, _, _ = preprocess_image(rgb_640, [], cfg)
        assert f.dtype == np.float32
