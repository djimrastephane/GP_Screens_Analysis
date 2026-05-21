"""Unit tests for src/detection modules."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.base import Detection, DetectionResult
from src.detection.nms import iou, nms
from src.detection.contour import ContourDetector, ContourConfig, _detect_dark_blobs, _detect_color_anomalies
from src.detection.store import ensure_table, upsert_detection_run, get_detection_runs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_detection(x1=10, y1=10, x2=60, y2=60, conf=0.8, cls=0, name="dark_blob", src="contour_dark"):
    area = int((x2 - x1) * (y2 - y1))
    return Detection(
        x1=x1, y1=y1, x2=x2, y2=y2,
        confidence=conf, class_id=cls, class_name=name,
        area_px=area, area_pct=area / (640 * 640) * 100,
        source=src,
    )


@pytest.fixture
def clean_mesh() -> np.ndarray:
    """Uniform grey 640x640 — simulates a flat background with no blobs."""
    return np.full((640, 640, 3), 180, dtype=np.uint8)


@pytest.fixture
def mesh_with_hole() -> np.ndarray:
    """Grey 640x640 with a large dark square patch — simulates an erosion hole."""
    img = np.full((640, 640, 3), 180, dtype=np.uint8)
    # Dark patch at centre — large enough to pass min_area_frac=0.002
    img[280:360, 280:360, :] = 20   # 80x80 = 6400 px ~ 1.6% of 640*640=409600
    return img


@pytest.fixture
def rust_patch() -> np.ndarray:
    """Grey 640x640 with an orange HSV patch — simulates corrosion."""
    img = np.full((640, 640, 3), 160, dtype=np.uint8)
    # Orange in RGB ≈ (200, 100, 30)
    img[250:370, 250:370, :] = [200, 100, 30]  # 120x120 = 14400 px ~ 3.5%
    return img


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "det_test.db"


# ---------------------------------------------------------------------------
# Detection dataclass
# ---------------------------------------------------------------------------

class TestDetection:
    def test_width_height(self):
        d = _make_detection(x1=10, y1=20, x2=50, y2=80)
        assert d.width == 40
        assert d.height == 60

    def test_aspect_ratio(self):
        d = _make_detection(x1=0, y1=0, x2=100, y2=50)
        assert d.aspect_ratio == pytest.approx(2.0)

    def test_to_dict_keys(self):
        d = _make_detection()
        keys = d.to_dict().keys()
        for k in ("x1", "y1", "x2", "y2", "confidence", "class_name", "source"):
            assert k in keys


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------

class TestIoU:
    def test_identical_boxes(self):
        a = _make_detection(0, 0, 100, 100)
        assert iou(a, a) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = _make_detection(0, 0, 50, 50)
        b = _make_detection(100, 100, 200, 200)
        assert iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = _make_detection(0, 0, 100, 100)
        b = _make_detection(50, 50, 150, 150)
        val = iou(a, b)
        assert 0 < val < 1


class TestNMS:
    def test_empty(self):
        assert nms([]) == []

    def test_single(self):
        d = _make_detection()
        assert nms([d]) == [d]

    def test_suppresses_overlapping_lower_conf(self):
        high = _make_detection(0, 0, 100, 100, conf=0.9)
        low = _make_detection(5, 5, 95, 95, conf=0.3)
        result = nms([high, low], iou_threshold=0.4)
        assert len(result) == 1
        assert result[0].confidence == pytest.approx(0.9)

    def test_keeps_non_overlapping(self):
        a = _make_detection(0, 0, 50, 50, conf=0.9)
        b = _make_detection(300, 300, 400, 400, conf=0.8)
        result = nms([a, b], iou_threshold=0.4)
        assert len(result) == 2

    def test_different_classes_not_suppressed(self):
        a = _make_detection(0, 0, 100, 100, conf=0.9, cls=0)
        b = _make_detection(5, 5, 95, 95, conf=0.8, cls=1)
        result = nms([a, b], iou_threshold=0.4)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# ContourDetector
# ---------------------------------------------------------------------------

class TestDarkBlobDetection:
    def test_no_blobs_on_uniform_image(self, clean_mesh):
        cfg = ContourConfig()
        blobs = _detect_dark_blobs(clean_mesh, cfg)
        assert len(blobs) == 0

    def test_detects_dark_patch(self, mesh_with_hole):
        cfg = ContourConfig(min_area_frac=0.001)
        blobs = _detect_dark_blobs(mesh_with_hole, cfg)
        assert len(blobs) >= 1

    def test_blob_coords_within_image(self, mesh_with_hole):
        cfg = ContourConfig(min_area_frac=0.001)
        h, w = mesh_with_hole.shape[:2]
        blobs = _detect_dark_blobs(mesh_with_hole, cfg)
        for x1, y1, x2, y2, _ in blobs:
            assert 0 <= x1 < x2 <= w
            assert 0 <= y1 < y2 <= h


class TestColorAnomalyDetection:
    def test_no_anomaly_on_grey_image(self, clean_mesh):
        cfg = ContourConfig()
        patches = _detect_color_anomalies(clean_mesh, cfg)
        assert len(patches) == 0

    def test_detects_orange_patch(self, rust_patch):
        cfg = ContourConfig(min_area_frac=0.001)
        patches = _detect_color_anomalies(rust_patch, cfg)
        assert len(patches) >= 1


class TestContourDetector:
    def test_returns_list(self, clean_mesh):
        det = ContourDetector()
        result = det.detect(clean_mesh)
        assert isinstance(result, list)

    def test_detects_dark_blob(self, mesh_with_hole):
        det = ContourDetector(ContourConfig(min_area_frac=0.001))
        result = det.detect(mesh_with_hole)
        sources = {d.source for d in result}
        assert "contour_dark" in sources

    def test_detects_color_anomaly(self, rust_patch):
        det = ContourDetector(ContourConfig(min_area_frac=0.001))
        result = det.detect(rust_patch)
        sources = {d.source for d in result}
        assert "contour_color" in sources

    def test_confidence_between_0_and_1(self, mesh_with_hole):
        det = ContourDetector(ContourConfig(min_area_frac=0.001))
        for d in det.detect(mesh_with_hole):
            assert 0.0 <= d.confidence <= 1.0

    def test_area_pct_plausible(self, mesh_with_hole):
        det = ContourDetector(ContourConfig(min_area_frac=0.001))
        for d in det.detect(mesh_with_hole):
            assert 0 < d.area_pct <= 20.0

    def test_disable_color_anomaly(self, rust_patch):
        cfg = ContourConfig(run_color_anomaly=False, min_area_frac=0.001)
        det = ContourDetector(cfg)
        result = det.detect(rust_patch)
        assert all(d.source != "contour_color" for d in result)

    def test_name_and_version(self):
        det = ContourDetector()
        assert det.name == "ContourDetector"
        assert det.version is not None

    def test_ready_always_true(self):
        assert ContourDetector().ready is True


# ---------------------------------------------------------------------------
# Database store
# ---------------------------------------------------------------------------

class TestDetectionStore:
    def _make_result(self, image_id="img_abc", n_dets=2) -> DetectionResult:
        from datetime import datetime, timezone
        dets = [_make_detection(i * 100, i * 100, i * 100 + 80, i * 100 + 80) for i in range(n_dets)]
        return DetectionResult(
            image_id=image_id,
            source_filename="test.jpg",
            detector_name="ContourDetector",
            model_version="1.0",
            detections=dets,
            image_width=640,
            image_height=640,
            run_timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def test_insert_and_retrieve(self, tmp_db):
        SessionFactory = ensure_table(tmp_db)
        result = self._make_result()
        with SessionFactory() as session:
            upsert_detection_run(session, result)
            session.commit()
        with SessionFactory() as session:
            runs = get_detection_runs(session)
        assert len(runs) == 1
        assert runs[0].detection_count == 2

    def test_detections_json_roundtrip(self, tmp_db):
        SessionFactory = ensure_table(tmp_db)
        result = self._make_result(n_dets=3)
        with SessionFactory() as session:
            upsert_detection_run(session, result)
            session.commit()
        with SessionFactory() as session:
            runs = get_detection_runs(session)
        dets = json.loads(runs[0].detections_json)
        assert len(dets) == 3
        assert "class_name" in dets[0]

    def test_upsert_replaces_existing(self, tmp_db):
        SessionFactory = ensure_table(tmp_db)
        r1 = self._make_result(n_dets=2)
        r2 = self._make_result(n_dets=5)
        with SessionFactory() as session:
            upsert_detection_run(session, r1)
            session.commit()
        with SessionFactory() as session:
            upsert_detection_run(session, r2)
            session.commit()
        with SessionFactory() as session:
            runs = get_detection_runs(session)
        assert len(runs) == 1
        assert runs[0].detection_count == 5

    def test_filter_by_image_id(self, tmp_db):
        SessionFactory = ensure_table(tmp_db)
        with SessionFactory() as session:
            upsert_detection_run(session, self._make_result(image_id="img_a"))
            upsert_detection_run(session, self._make_result(image_id="img_b"))
            session.commit()
        with SessionFactory() as session:
            runs = get_detection_runs(session, image_id="img_a")
        assert len(runs) == 1
        assert runs[0].image_id == "img_a"
