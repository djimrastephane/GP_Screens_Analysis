"""Unit tests for src/segmentation modules."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.segmentation.metrics import (
    bbox_area_px, fill_ratio, mask_area_pct, mask_area_px, union_mask
)
from src.segmentation.contour_mask import ContourMaskSegmenter
from src.segmentation.grabcut import GrabCutSegmenter
from src.segmentation.overlay import draw_overlay
from src.segmentation.base import SegmentationMask, SegmentationResult
from src.segmentation.store import ensure_table, upsert_segmentation_run, get_segmentation_runs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def grey_img() -> np.ndarray:
    return np.full((640, 640, 3), 180, dtype=np.uint8)


@pytest.fixture
def img_with_dark_blob() -> np.ndarray:
    img = np.full((640, 640, 3), 200, dtype=np.uint8)
    img[280:360, 270:370, :] = 15  # 80x100 dark patch
    return img


@pytest.fixture
def img_with_rust() -> np.ndarray:
    img = np.full((640, 640, 3), 170, dtype=np.uint8)
    img[250:370, 250:370, :] = [200, 100, 30]  # orange patch
    return img


@pytest.fixture
def binary_mask_partial() -> np.ndarray:
    m = np.zeros((640, 640), dtype=np.uint8)
    m[100:200, 100:200] = 255  # 100x100 = 10000 px
    return m


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "seg_test.db"


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

class TestMaskAreaPx:
    def test_zeros(self):
        assert mask_area_px(np.zeros((100, 100), dtype=np.uint8)) == 0

    def test_full(self):
        assert mask_area_px(np.full((100, 100), 255, dtype=np.uint8)) == 10000

    def test_partial(self, binary_mask_partial):
        assert mask_area_px(binary_mask_partial) == 10000


class TestMaskAreaPct:
    def test_full_mask(self):
        assert mask_area_pct(np.full((640, 640), 255, dtype=np.uint8)) == pytest.approx(100.0)

    def test_zero_mask(self):
        assert mask_area_pct(np.zeros((640, 640), dtype=np.uint8)) == pytest.approx(0.0)

    def test_partial(self, binary_mask_partial):
        expected = 10000 / (640 * 640) * 100
        assert mask_area_pct(binary_mask_partial) == pytest.approx(expected, rel=1e-4)


class TestBboxAreaPx:
    def test_basic(self):
        assert bbox_area_px(10, 20, 110, 120) == 10000

    def test_inverted(self):
        assert bbox_area_px(100, 100, 50, 50) == 0


class TestFillRatio:
    def test_full_fill(self):
        m = np.zeros((640, 640), dtype=np.uint8)
        m[100:200, 100:200] = 255
        ratio = fill_ratio(m, 100, 100, 200, 200)
        assert ratio == pytest.approx(1.0, rel=0.01)

    def test_empty_mask(self):
        m = np.zeros((640, 640), dtype=np.uint8)
        assert fill_ratio(m, 100, 100, 200, 200) == pytest.approx(0.0)

    def test_zero_area_box(self):
        m = np.zeros((640, 640), dtype=np.uint8)
        assert fill_ratio(m, 100, 100, 100, 100) == pytest.approx(0.0)


class TestUnionMask:
    def test_union_two_non_overlapping(self):
        a = np.zeros((100, 100), dtype=np.uint8)
        b = np.zeros((100, 100), dtype=np.uint8)
        a[0:50, :] = 255
        b[50:, :] = 255
        u = union_mask([a, b])
        assert mask_area_px(u) == 10000

    def test_union_identical(self):
        m = np.full((100, 100), 255, dtype=np.uint8)
        u = union_mask([m, m])
        assert mask_area_px(u) == mask_area_px(m)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            union_mask([])


# ---------------------------------------------------------------------------
# ContourMaskSegmenter
# ---------------------------------------------------------------------------

class TestContourMaskSegmenter:
    def test_output_shape(self, img_with_dark_blob):
        seg = ContourMaskSegmenter()
        mask = seg.segment_one(img_with_dark_blob, 260, 270, 380, 370, "dark_blob")
        assert mask.shape == (640, 640)

    def test_output_dtype(self, img_with_dark_blob):
        seg = ContourMaskSegmenter()
        mask = seg.segment_one(img_with_dark_blob, 260, 270, 380, 370, "dark_blob")
        assert mask.dtype == np.uint8

    def test_only_0_or_255(self, img_with_dark_blob):
        seg = ContourMaskSegmenter()
        mask = seg.segment_one(img_with_dark_blob, 260, 270, 380, 370, "dark_blob")
        unique = set(np.unique(mask))
        assert unique.issubset({0, 255})

    def test_nonzero_pixels_in_blob_region(self, img_with_dark_blob):
        seg = ContourMaskSegmenter()
        mask = seg.segment_one(img_with_dark_blob, 260, 270, 380, 370, "dark_blob")
        # The blob (280:360, 270:370) should have some foreground pixels
        roi = mask[280:360, 270:370]
        assert (roi > 0).sum() > 0

    def test_color_anomaly_detects_rust(self, img_with_rust):
        seg = ContourMaskSegmenter()
        mask = seg.segment_one(img_with_rust, 240, 240, 380, 380, "color_anomaly")
        assert mask_area_px(mask) > 0

    def test_uniform_image_mostly_empty(self, grey_img):
        seg = ContourMaskSegmenter()
        mask = seg.segment_one(grey_img, 100, 100, 300, 300, "dark_blob")
        # Uniform image should produce few or no foreground pixels
        assert mask_area_pct(mask) < 5.0

    def test_name_version(self):
        seg = ContourMaskSegmenter()
        assert seg.name == "ContourMaskSegmenter"
        assert seg.version is not None
        assert seg.ready is True


# ---------------------------------------------------------------------------
# GrabCutSegmenter
# ---------------------------------------------------------------------------

class TestGrabCutSegmenter:
    def test_output_shape(self, img_with_dark_blob):
        seg = GrabCutSegmenter()
        mask = seg.segment_one(img_with_dark_blob, 260, 270, 380, 370, "dark_blob")
        assert mask.shape == (640, 640)

    def test_output_dtype(self, img_with_dark_blob):
        seg = GrabCutSegmenter()
        mask = seg.segment_one(img_with_dark_blob, 260, 270, 380, 370, "dark_blob")
        assert mask.dtype == np.uint8

    def test_only_0_or_255(self, img_with_dark_blob):
        seg = GrabCutSegmenter()
        mask = seg.segment_one(img_with_dark_blob, 260, 270, 380, 370, "dark_blob")
        unique = set(np.unique(mask))
        assert unique.issubset({0, 255})

    def test_tiny_box_falls_back(self, img_with_dark_blob):
        # Box smaller than _MIN_DIM (30px) — should not crash, falls back
        seg = GrabCutSegmenter()
        mask = seg.segment_one(img_with_dark_blob, 280, 280, 295, 295, "dark_blob")
        assert mask.shape == (640, 640)

    def test_name_version(self):
        seg = GrabCutSegmenter()
        assert seg.name == "GrabCutSegmenter"
        assert seg.ready is True


# ---------------------------------------------------------------------------
# overlay
# ---------------------------------------------------------------------------

class TestDrawOverlay:
    def test_output_shape(self, img_with_dark_blob):
        mask = np.zeros((640, 640), dtype=np.uint8)
        mask[280:360, 270:370] = 255
        det = {"x1": 270, "y1": 280, "x2": 370, "y2": 360,
               "class_name": "dark_blob", "confidence": 0.7}
        out = draw_overlay(img_with_dark_blob, [(mask, det)])
        assert out.shape == img_with_dark_blob.shape

    def test_output_dtype(self, img_with_dark_blob):
        out = draw_overlay(img_with_dark_blob, [])
        assert out.dtype == np.uint8

    def test_no_detections_returns_image(self, grey_img):
        out = draw_overlay(grey_img, [])
        assert out.shape == grey_img.shape


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------

class TestSegmentationStore:
    def _make_result(self, image_id: str = "img_xyz") -> SegmentationResult:
        from datetime import datetime, timezone
        masks = [
            SegmentationMask(
                detection_index=i, class_id=0, class_name="dark_blob",
                detection_confidence=0.7, source="ContourMaskSegmenter",
                mask=np.zeros((640, 640), dtype=np.uint8),
                defect_area_px=500, defect_area_pct=0.12,
                bbox_area_px=6400, fill_ratio=0.078,
            )
            for i in range(3)
        ]
        return SegmentationResult(
            image_id=image_id, source_filename="test.jpg",
            segmenter_name="ContourMaskSegmenter", model_version="1.0",
            masks=masks, image_width=640, image_height=640,
            run_timestamp=datetime.now(timezone.utc).isoformat(),
            composite_defect_area_px=1200, composite_defect_area_pct=0.29,
        )

    def test_insert_and_retrieve(self, tmp_db):
        SessionFactory = ensure_table(tmp_db)
        result = self._make_result()
        with SessionFactory() as session:
            upsert_segmentation_run(session, result)
            session.commit()
        with SessionFactory() as session:
            runs = get_segmentation_runs(session)
        assert len(runs) == 1
        assert runs[0].n_masks == 3

    def test_composite_area_persisted(self, tmp_db):
        SessionFactory = ensure_table(tmp_db)
        with SessionFactory() as session:
            upsert_segmentation_run(session, self._make_result())
            session.commit()
        with SessionFactory() as session:
            runs = get_segmentation_runs(session)
        assert runs[0].composite_defect_area_pct == pytest.approx(0.29, rel=0.01)

    def test_per_mask_json_roundtrip(self, tmp_db):
        SessionFactory = ensure_table(tmp_db)
        with SessionFactory() as session:
            upsert_segmentation_run(session, self._make_result())
            session.commit()
        with SessionFactory() as session:
            runs = get_segmentation_runs(session)
        per_mask = json.loads(runs[0].per_mask_json)
        assert len(per_mask) == 3
        assert "class_name" in per_mask[0]

    def test_upsert_replaces(self, tmp_db):
        SessionFactory = ensure_table(tmp_db)
        r1 = self._make_result()
        r1.composite_defect_area_pct = 1.0
        r2 = self._make_result()
        r2.composite_defect_area_pct = 5.0
        with SessionFactory() as session:
            upsert_segmentation_run(session, r1)
            session.commit()
        with SessionFactory() as session:
            upsert_segmentation_run(session, r2)
            session.commit()
        with SessionFactory() as session:
            runs = get_segmentation_runs(session)
        assert len(runs) == 1
        assert runs[0].composite_defect_area_pct == pytest.approx(5.0, rel=0.01)
