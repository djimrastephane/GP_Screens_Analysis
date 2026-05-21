"""Unit tests for src/annotation modules."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.annotation.colours import (
    FAILURE_TYPE_RGB, SEVERITY_THICKNESS,
    get_colour, get_severity_colour, rgb_to_bgr,
)
from src.annotation.draw import (
    draw_header_bar, draw_legend, draw_box, overlay_mask,
)
from src.annotation.composer import compose_annotated, compose_panels
from src.annotation.store import ensure_table, upsert_annotation, get_all_annotations


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def blank_rgb() -> np.ndarray:
    return np.full((640, 640, 3), 180, dtype=np.uint8)


@pytest.fixture
def small_rgb() -> np.ndarray:
    return np.full((480, 640, 3), 180, dtype=np.uint8)


@pytest.fixture
def mask_centre() -> np.ndarray:
    m = np.zeros((640, 640), dtype=np.uint8)
    m[280:360, 280:360] = 255
    return m


@pytest.fixture
def sample_det():
    return {"x1": 280, "y1": 280, "x2": 360, "y2": 360,
            "class_name": "dark_blob", "confidence": 0.75, "source": "contour_dark"}


@pytest.fixture
def sample_defect():
    return {
        "detection_index": 0, "failure_type": "erosion_hole", "severity": "medium",
        "confidence": 0.75, "requires_human_review": False,
        "defect_area_px": 6400, "defect_area_pct_of_screen": 2.1,
        "equivalent_diameter_px": 90.3, "fill_ratio": 0.85,
        "equivalent_diameter_mm": None,
    }


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "ann_test.db"


# ---------------------------------------------------------------------------
# colours
# ---------------------------------------------------------------------------

class TestColours:
    def test_all_failure_types_have_colour(self):
        for ft in [
            "erosion_hole", "corrosion_pitting", "wire_wrap_failure",
            "screen_collapse", "mechanical_damage",
            "plugging_partial", "plugging_complete", "unknown",
        ]:
            col = get_colour(ft)
            assert len(col) == 3
            assert all(0 <= c <= 255 for c in col)

    def test_unknown_fallback(self):
        col = get_colour("nonexistent_type")
        assert col == FAILURE_TYPE_RGB["unknown"]

    def test_all_severities_have_colour(self):
        for sev in ["low", "medium", "high", "critical"]:
            col = get_severity_colour(sev)
            assert len(col) == 3

    def test_all_severities_have_thickness(self):
        for sev in ["low", "medium", "high", "critical"]:
            t = SEVERITY_THICKNESS[sev]
            assert isinstance(t, int) and t >= 1

    def test_thickness_increases_with_severity(self):
        t = [SEVERITY_THICKNESS[s] for s in ["low", "medium", "high", "critical"]]
        assert t == sorted(t)

    def test_rgb_to_bgr(self):
        assert rgb_to_bgr((10, 20, 30)) == (30, 20, 10)

    def test_rgb_to_bgr_roundtrip(self):
        col = (100, 150, 200)
        assert rgb_to_bgr(rgb_to_bgr(col)) == col


# ---------------------------------------------------------------------------
# draw
# ---------------------------------------------------------------------------

class TestOverlayMask:
    def test_output_shape(self, blank_rgb, mask_centre):
        out = overlay_mask(blank_rgb, mask_centre, (220, 55, 55))
        assert out.shape == blank_rgb.shape

    def test_dtype_uint8(self, blank_rgb, mask_centre):
        out = overlay_mask(blank_rgb, mask_centre, (220, 55, 55))
        assert out.dtype == np.uint8

    def test_masked_pixels_changed(self, blank_rgb, mask_centre):
        out = overlay_mask(blank_rgb, mask_centre, (220, 55, 55))
        roi_before = blank_rgb[280:360, 280:360, 0].mean()
        roi_after = out[280:360, 280:360, 0].mean()
        assert roi_before != roi_after

    def test_unmasked_pixels_unchanged(self, blank_rgb, mask_centre):
        out = overlay_mask(blank_rgb.copy(), mask_centre, (220, 55, 55))
        np.testing.assert_array_equal(out[0:10, 0:10], blank_rgb[0:10, 0:10])

    def test_empty_mask_unchanged(self, blank_rgb):
        empty = np.zeros((640, 640), dtype=np.uint8)
        out = overlay_mask(blank_rgb.copy(), empty, (220, 55, 55))
        np.testing.assert_array_equal(out, blank_rgb)


class TestDrawBox:
    def test_does_not_crash(self, blank_rgb):
        import cv2
        bgr = cv2.cvtColor(blank_rgb, cv2.COLOR_RGB2BGR)
        draw_box(bgr, 100, 100, 200, 200, (220, 55, 55), thickness=2)

    def test_dashed_does_not_crash(self, blank_rgb):
        import cv2
        bgr = cv2.cvtColor(blank_rgb, cv2.COLOR_RGB2BGR)
        draw_box(bgr, 100, 100, 200, 200, (220, 55, 55), thickness=1, dashed=True)


class TestDrawHeaderBar:
    def test_shape(self):
        bar = draw_header_bar("test.jpg", 25.0, 10, "medium", 2, 640)
        assert bar.shape[1] == 640
        assert bar.dtype == np.uint8

    def test_height_matches_constant(self):
        from src.annotation.colours import HEADER_H
        bar = draw_header_bar("test.jpg", 12.0, 5, "low", 0, 640)
        assert bar.shape[0] == HEADER_H

    def test_rgb_output(self):
        bar = draw_header_bar("test.jpg", 30.0, 8, "high", 1, 640)
        assert bar.shape[2] == 3


class TestDrawLegend:
    def test_does_not_crash(self, blank_rgb):
        pil = Image.fromarray(blank_rgb)
        draw_legend(pil, ["erosion_hole", "corrosion_pitting"], {"erosion_hole": 3, "corrosion_pitting": 5})

    def test_empty_list_no_crash(self, blank_rgb):
        pil = Image.fromarray(blank_rgb)
        draw_legend(pil, [], {})


# ---------------------------------------------------------------------------
# composer
# ---------------------------------------------------------------------------

class TestComposeAnnotated:
    def test_output_dtype(self, blank_rgb, sample_det, sample_defect, mask_centre):
        out = compose_annotated(
            blank_rgb, [sample_det], [sample_defect], [mask_centre],
            "test.jpg", 25.0, "medium", 0,
        )
        assert out.dtype == np.uint8

    def test_header_adds_rows(self, blank_rgb, sample_det, sample_defect, mask_centre):
        from src.annotation.colours import HEADER_H
        out = compose_annotated(
            blank_rgb, [sample_det], [sample_defect], [mask_centre],
            "test.jpg", 25.0, "medium", 0, include_header=True,
        )
        assert out.shape[0] == blank_rgb.shape[0] + HEADER_H

    def test_no_header_same_height(self, blank_rgb, sample_det, sample_defect, mask_centre):
        out = compose_annotated(
            blank_rgb, [sample_det], [sample_defect], [mask_centre],
            "test.jpg", 25.0, "medium", 0, include_header=False,
        )
        assert out.shape[0] == blank_rgb.shape[0]

    def test_width_preserved(self, blank_rgb, sample_det, sample_defect, mask_centre):
        out = compose_annotated(
            blank_rgb, [sample_det], [sample_defect], [mask_centre],
            "test.jpg", 25.0, "medium", 0,
        )
        assert out.shape[1] == blank_rgb.shape[1]

    def test_empty_detections(self, blank_rgb):
        out = compose_annotated(blank_rgb, [], [], [], "test.jpg", 5.0, "low", 0)
        assert out.dtype == np.uint8

    def test_review_flag_does_not_crash(self, blank_rgb, sample_det, mask_centre):
        defect_review = {
            "detection_index": 0, "failure_type": "corrosion_pitting", "severity": "high",
            "confidence": 0.45, "requires_human_review": True,
            "defect_area_px": 3000, "defect_area_pct_of_screen": 1.0,
            "equivalent_diameter_px": 61.8, "fill_ratio": 0.6, "equivalent_diameter_mm": None,
        }
        out = compose_annotated(
            blank_rgb, [sample_det], [defect_review], [mask_centre],
            "test.jpg", 10.0, "high", 1,
        )
        assert out.dtype == np.uint8


class TestComposePanels:
    def test_output_width_is_3x(self, blank_rgb, mask_centre):
        annotated = blank_rgb.copy()
        panel = compose_panels(blank_rgb, annotated, mask_centre)
        assert panel.shape[1] == blank_rgb.shape[1] * 3

    def test_output_height_matches(self, blank_rgb, mask_centre):
        annotated = blank_rgb.copy()
        panel = compose_panels(blank_rgb, annotated, mask_centre)
        assert panel.shape[0] == blank_rgb.shape[0]

    def test_with_header(self, blank_rgb, mask_centre):
        from src.annotation.colours import HEADER_H
        annotated = blank_rgb.copy()
        header = draw_header_bar("t.jpg", 10.0, 5, "low", 0, blank_rgb.shape[1] * 3)
        panel = compose_panels(blank_rgb, annotated, mask_centre, header=header)
        assert panel.shape[0] == blank_rgb.shape[0] + HEADER_H

    def test_dtype(self, blank_rgb, mask_centre):
        panel = compose_panels(blank_rgb, blank_rgb.copy(), mask_centre)
        assert panel.dtype == np.uint8


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------

class TestAnnotationStore:
    def test_insert_and_retrieve(self, tmp_db):
        SessionFactory = ensure_table(tmp_db)
        with SessionFactory() as session:
            upsert_annotation(session, "img_a", "a.jpg", "/out/a.png", None, "2026-05-21T00:00:00Z")
            session.commit()
        with SessionFactory() as session:
            recs = get_all_annotations(session)
        assert len(recs) == 1
        assert recs[0].annotated_path == "/out/a.png"

    def test_upsert_replaces(self, tmp_db):
        SessionFactory = ensure_table(tmp_db)
        with SessionFactory() as session:
            upsert_annotation(session, "img_a", "a.jpg", "/old.png", None, "2026-05-21T00:00:00Z")
            session.commit()
        with SessionFactory() as session:
            upsert_annotation(session, "img_a", "a.jpg", "/new.png", "/panel.png", "2026-05-22T00:00:00Z")
            session.commit()
        with SessionFactory() as session:
            recs = get_all_annotations(session)
        assert len(recs) == 1
        assert recs[0].annotated_path == "/new.png"
        assert recs[0].panel_path == "/panel.png"

    def test_panel_path_nullable(self, tmp_db):
        SessionFactory = ensure_table(tmp_db)
        with SessionFactory() as session:
            upsert_annotation(session, "img_b", "b.jpg", "/out/b.png", None, "2026-05-21T00:00:00Z")
            session.commit()
        with SessionFactory() as session:
            recs = get_all_annotations(session)
        assert recs[0].panel_path is None
