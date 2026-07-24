"""Unit tests for src/classification modules."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classification.base import FailureType, FeatureVector, ClassificationResult
from src.classification.severity import area_to_severity, worst_severity
from src.classification.features import extract_features
from src.classification.rules import classify_features
from src.classification.classifier import RuleBasedClassifier
from src.classification.store import (
    ensure_table, upsert_classification_run, get_classification_runs,
    set_reviewed, get_review_status,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _fv(
    circ=0.5, sol=0.85, ar=1.0, ext=0.5, area=0.5,
    hue=120.0, std_hue=10.0, sat=50.0, val=50.0, src="contour_dark",
) -> FeatureVector:
    return FeatureVector(
        circularity=circ, solidity=sol, aspect_ratio=ar, extent=ext,
        area_pct=area, mean_hue=hue, std_hue=std_hue,
        mean_saturation=sat, mean_value=val, detection_source=src,
    )


def _det(x1=100, y1=100, x2=200, y2=200, cls="dark_blob", src="contour_dark", conf=0.7):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "class_name": cls, "source": src, "confidence": conf,
            "class_id": 0, "area_pct": 1.5, "detection_index": 0}


@pytest.fixture
def img_with_dark_blob() -> np.ndarray:
    img = np.full((640, 640, 3), 200, dtype=np.uint8)
    img[280:360, 270:370, :] = 15
    return img


@pytest.fixture
def img_with_rust() -> np.ndarray:
    img = np.full((640, 640, 3), 170, dtype=np.uint8)
    img[250:370, 250:370, :] = [200, 100, 30]
    return img


@pytest.fixture
def mask_blob() -> np.ndarray:
    m = np.zeros((640, 640), dtype=np.uint8)
    m[280:360, 270:370] = 255
    return m


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "cls_test.db"


# ---------------------------------------------------------------------------
# severity
# ---------------------------------------------------------------------------

class TestAreaToSeverity:
    def test_low(self):
        assert area_to_severity(2.0) == "low"

    def test_medium(self):
        assert area_to_severity(10.0) == "medium"

    def test_high(self):
        assert area_to_severity(30.0) == "high"

    def test_critical(self):
        assert area_to_severity(60.0) == "critical"

    def test_boundary_low_medium(self):
        assert area_to_severity(5.0) == "medium"

    def test_screen_collapse_escalates(self):
        # 3% area → normally "low", but screen_collapse escalates to "medium"
        assert area_to_severity(3.0, "screen_collapse") == "medium"

    def test_plugging_complete_escalates(self):
        # 10% area → "medium", escalates to "high"
        assert area_to_severity(10.0, "plugging_complete") == "high"

    def test_critical_does_not_escalate_beyond(self):
        assert area_to_severity(60.0, "screen_collapse") == "critical"


class TestWorstSeverity:
    def test_single(self):
        assert worst_severity(["medium"]) == "medium"

    def test_picks_highest(self):
        assert worst_severity(["low", "high", "medium"]) == "high"

    def test_empty_returns_low(self):
        assert worst_severity([]) == "low"

    def test_critical_dominates(self):
        assert worst_severity(["low", "critical", "medium"]) == "critical"


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

class TestExtractFeatures:
    def test_returns_feature_vector(self, img_with_dark_blob, mask_blob):
        det = _det()
        fv = extract_features(img_with_dark_blob, mask_blob, det)
        assert isinstance(fv, FeatureVector)

    def test_circularity_between_0_and_1(self, img_with_dark_blob, mask_blob):
        fv = extract_features(img_with_dark_blob, mask_blob, _det())
        assert 0.0 <= fv.circularity <= 1.0

    def test_solidity_between_0_and_1(self, img_with_dark_blob, mask_blob):
        fv = extract_features(img_with_dark_blob, mask_blob, _det())
        assert 0.0 <= fv.solidity <= 1.0

    def test_empty_mask_does_not_crash(self, img_with_dark_blob):
        empty = np.zeros((640, 640), dtype=np.uint8)
        fv = extract_features(img_with_dark_blob, empty, _det())
        assert isinstance(fv, FeatureVector)

    def test_rust_image_high_saturation(self, img_with_rust):
        m = np.zeros((640, 640), dtype=np.uint8)
        m[250:370, 250:370] = 255
        fv = extract_features(img_with_rust, m, _det(x1=250, y1=250, x2=370, y2=370))
        assert fv.mean_saturation > 60

    def test_rust_image_orange_hue(self, img_with_rust):
        m = np.zeros((640, 640), dtype=np.uint8)
        m[250:370, 250:370] = 255
        fv = extract_features(img_with_rust, m, _det(x1=250, y1=250, x2=370, y2=370))
        assert 5 <= fv.mean_hue <= 35


# ---------------------------------------------------------------------------
# rules
# ---------------------------------------------------------------------------

class TestClassifyFeatures:
    def test_corrosion_from_orange_hue(self):
        fv = _fv(hue=20, sat=120, val=80, src="contour_dark")
        result = classify_features(fv, 0, "dark_blob", "medium")
        assert result.failure_type == FailureType.CORROSION_PITTING

    def test_corrosion_from_color_source(self):
        fv = _fv(hue=80, sat=40, src="contour_color")
        result = classify_features(fv, 0, "color_anomaly", "low")
        assert result.failure_type == FailureType.CORROSION_PITTING

    def test_plugging_partial_from_high_ar(self):
        fv = _fv(ar=5.0, circ=0.2, hue=80, sat=30)
        result = classify_features(fv, 0, "dark_blob", "medium")
        assert result.failure_type == FailureType.PLUGGING_PARTIAL

    def test_screen_collapse_from_large_low_solidity(self):
        fv = _fv(area=6.0, sol=0.35, hue=80, sat=20)
        result = classify_features(fv, 0, "dark_blob", "medium")
        assert result.failure_type == FailureType.SCREEN_COLLAPSE

    def test_wire_wrap_from_elongated_low_circ(self):
        fv = _fv(ar=2.5, circ=0.20, hue=80, sat=20, area=1.0)
        result = classify_features(fv, 0, "dark_blob", "medium")
        assert result.failure_type == FailureType.WIRE_WRAP_FAILURE

    def test_erosion_hole_from_compact_blob(self):
        fv = _fv(circ=0.55, ar=1.1, hue=80, sat=20, val=20, area=0.5)
        result = classify_features(fv, 0, "dark_blob", "low")
        assert result.failure_type == FailureType.EROSION_HOLE

    def test_confidence_between_0_and_1(self):
        fv = _fv()
        result = classify_features(fv, 0, "dark_blob", "low")
        assert 0.0 <= result.confidence <= 1.0

    def test_reasoning_is_not_empty(self):
        fv = _fv()
        result = classify_features(fv, 0, "dark_blob", "low")
        assert len(result.reasoning) > 0

    def test_review_flag_on_low_confidence(self):
        # mechanical damage fallback has conf=0.42 < 0.65
        fv = _fv(circ=0.1, ar=1.2, hue=80, sat=20, area=0.2, val=150)
        result = classify_features(fv, 0, "dark_blob", "low", review_threshold=0.65)
        assert result.requires_human_review is True


# ---------------------------------------------------------------------------
# RuleBasedClassifier
# ---------------------------------------------------------------------------

class TestRuleBasedClassifier:
    def test_classify_one_returns_result(self, img_with_dark_blob, mask_blob):
        clf = RuleBasedClassifier()
        det = _det()
        result = clf.classify_one(img_with_dark_blob, mask_blob, det, composite_defect_area_pct=5.0)
        assert isinstance(result, ClassificationResult)

    def test_classify_image_returns_all_detections(self, img_with_dark_blob):
        clf = RuleBasedClassifier()
        masks = [np.zeros((640, 640), dtype=np.uint8) for _ in range(3)]
        dets = [_det(x1=i*100, x2=i*100+50) for i in range(3)]
        from datetime import datetime, timezone
        result = clf.classify_image(
            "img_x", "test.jpg", img_with_dark_blob, dets, masks, 5.0,
            datetime.now(timezone.utc).isoformat()
        )
        assert result.count == 3

    def test_dominant_failure_type_set(self, img_with_dark_blob, mask_blob):
        clf = RuleBasedClassifier()
        masks = [mask_blob] * 3
        dets = [_det() for _ in range(3)]
        from datetime import datetime, timezone
        result = clf.classify_image(
            "img_y", "test.jpg", img_with_dark_blob, dets, masks, 5.0,
            datetime.now(timezone.utc).isoformat()
        )
        assert result.dominant_failure_type in [ft.value for ft in FailureType]


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------

class TestClassificationStore:
    def _make_result(self, image_id="img_abc"):
        from src.classification.base import ImageClassificationResult, ClassificationResult, FeatureVector, FailureType
        from datetime import datetime, timezone
        fv = _fv()
        cls = ClassificationResult(
            detection_index=0, class_name="dark_blob",
            failure_type=FailureType.EROSION_HOLE, confidence=0.75,
            severity="medium", requires_human_review=False,
            reasoning="test", features=fv,
        )
        return ImageClassificationResult(
            image_id=image_id, source_filename="test.jpg",
            classifier_name="RuleBasedClassifier",
            classifications=[cls],
            run_timestamp=datetime.now(timezone.utc).isoformat(),
            dominant_failure_type="erosion_hole",
            composite_defect_area_pct=5.0,
            overall_severity="medium",
        )

    def test_insert_and_retrieve(self, tmp_db):
        SessionFactory = ensure_table(tmp_db)
        result = self._make_result()
        with SessionFactory() as session:
            upsert_classification_run(session, result)
            session.commit()
        with SessionFactory() as session:
            runs = get_classification_runs(session)
        assert len(runs) == 1
        assert runs[0].dominant_failure_type == "erosion_hole"

    def test_json_roundtrip(self, tmp_db):
        SessionFactory = ensure_table(tmp_db)
        with SessionFactory() as session:
            upsert_classification_run(session, self._make_result())
            session.commit()
        with SessionFactory() as session:
            runs = get_classification_runs(session)
        data = json.loads(runs[0].classifications_json)
        assert len(data) == 1
        assert "failure_type" in data[0]
        assert "features" in data[0]

    def test_upsert_replaces(self, tmp_db):
        SessionFactory = ensure_table(tmp_db)
        r1 = self._make_result()
        r1.overall_severity = "low"
        r2 = self._make_result()
        r2.overall_severity = "critical"
        with SessionFactory() as session:
            upsert_classification_run(session, r1)
            session.commit()
        with SessionFactory() as session:
            upsert_classification_run(session, r2)
            session.commit()
        with SessionFactory() as session:
            runs = get_classification_runs(session)
        assert len(runs) == 1
        assert runs[0].overall_severity == "critical"


# ---------------------------------------------------------------------------
# review status
# ---------------------------------------------------------------------------

class TestReviewStatus:
    def test_set_reviewed_creates_record(self, tmp_db):
        SessionFactory = ensure_table(tmp_db)
        with SessionFactory() as session:
            set_reviewed(session, "img_a", 0, reviewed=True, note="checked")
            session.commit()
        with SessionFactory() as session:
            rows = get_review_status(session)
        assert len(rows) == 1
        assert rows[0].id == "img_a__0"
        assert rows[0].reviewed is True
        assert rows[0].note == "checked"
        assert rows[0].reviewed_at is not None

    def test_set_reviewed_updates_existing(self, tmp_db):
        SessionFactory = ensure_table(tmp_db)
        with SessionFactory() as session:
            set_reviewed(session, "img_a", 0, reviewed=True)
            session.commit()
        with SessionFactory() as session:
            set_reviewed(session, "img_a", 0, reviewed=False)
            session.commit()
        with SessionFactory() as session:
            rows = get_review_status(session)
        assert len(rows) == 1
        assert rows[0].reviewed is False
        assert rows[0].reviewed_at is None

    def test_filter_by_image_id(self, tmp_db):
        SessionFactory = ensure_table(tmp_db)
        with SessionFactory() as session:
            set_reviewed(session, "img_a", 0, reviewed=True)
            set_reviewed(session, "img_b", 0, reviewed=True)
            session.commit()
        with SessionFactory() as session:
            rows = get_review_status(session, image_id="img_a")
        assert len(rows) == 1
        assert rows[0].image_id == "img_a"
