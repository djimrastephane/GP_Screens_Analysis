"""Unit tests for src/quantification modules."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.quantification.measurements import (
    corrected_erosion_pct,
    defect_density,
    equivalent_diameter_px,
    pixels_to_mm,
    screen_region_area_px,
)
from src.quantification.aggregator import aggregate, AggregatedMetrics
from src.quantification.report import (
    DefectMeasurement, QuantificationReport, build_report, export_csv
)
from src.quantification.store import ensure_table, get_all_reports, upsert_report


# ---------------------------------------------------------------------------
# measurements
# ---------------------------------------------------------------------------

class TestEquivalentDiameter:
    def test_zero_area(self):
        assert equivalent_diameter_px(0) == 0.0

    def test_circle_area(self):
        # Circle with diameter 10 → area = π×25 ≈ 78.54
        area = math.pi * 25
        diam = equivalent_diameter_px(area)
        assert abs(diam - 10.0) < 0.01

    def test_positive_area(self):
        assert equivalent_diameter_px(1000) > 0


class TestPixelsToMm:
    def test_basic(self):
        assert pixels_to_mm(100.0, 10.0) == pytest.approx(10.0)

    def test_zero_scale(self):
        assert pixels_to_mm(100.0, 0.0) == pytest.approx(0.0)

    def test_fractional(self):
        assert pixels_to_mm(35.7, 3.57) == pytest.approx(10.0, rel=1e-3)


class TestScreenRegionArea:
    def test_no_padding(self):
        assert screen_region_area_px(640, 640, 0, 0, 0, 0) == 640 * 640

    def test_symmetric_padding(self):
        # 80px top and bottom → content height = 480
        assert screen_region_area_px(640, 640, 80, 80, 0, 0) == 640 * 480

    def test_all_padding(self):
        # Extreme: more padding than image
        assert screen_region_area_px(100, 100, 60, 60, 0, 0) == 0

    def test_asymmetric_padding(self):
        result = screen_region_area_px(640, 640, 79, 79, 0, 0)
        assert result == 640 * 482


class TestCorrectedErosionPct:
    def test_zero_defect(self):
        assert corrected_erosion_pct(0, 100000) == pytest.approx(0.0)

    def test_full_coverage(self):
        assert corrected_erosion_pct(100000, 100000) == pytest.approx(100.0)

    def test_capped_at_100(self):
        assert corrected_erosion_pct(200000, 100000) == pytest.approx(100.0)

    def test_partial(self):
        result = corrected_erosion_pct(30000, 100000)
        assert result == pytest.approx(30.0)

    def test_zero_screen_area(self):
        assert corrected_erosion_pct(1000, 0) == pytest.approx(0.0)


class TestDefectDensity:
    def test_basic(self):
        result = defect_density(10, 100000)
        assert result == pytest.approx(1.0)   # 10 / 100000 * 10000

    def test_zero_area(self):
        assert defect_density(10, 0) == pytest.approx(0.0)

    def test_zero_defects(self):
        assert defect_density(0, 100000) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# aggregator
# ---------------------------------------------------------------------------

def _make_classifications(n=3, failure_type="erosion_hole", severity="medium"):
    return [
        {
            "detection_index": i,
            "failure_type": failure_type,
            "severity": severity,
            "confidence": 0.7,
            "requires_human_review": False,
        }
        for i in range(n)
    ]


def _make_seg_masks(n=3, area_px=5000, fill_ratio=0.75):
    return [
        {"detection_index": i, "defect_area_px": area_px, "fill_ratio": fill_ratio}
        for i in range(n)
    ]


class TestAggregate:
    def test_returns_aggregated_metrics(self):
        result = aggregate(
            _make_classifications(3), _make_seg_masks(3),
            composite_defect_area_px=12000,
            screen_area_px=300000,
            overall_severity="medium",
            dominant_failure_type="erosion_hole",
        )
        assert isinstance(result, AggregatedMetrics)

    def test_n_defects(self):
        result = aggregate(
            _make_classifications(5), _make_seg_masks(5),
            15000, 300000, "medium", "erosion_hole"
        )
        assert result.n_defects == 5

    def test_failure_type_count(self):
        cls = _make_classifications(4, "corrosion_pitting")
        result = aggregate(cls, _make_seg_masks(4), 10000, 300000, "medium", "corrosion_pitting")
        assert result.failure_type_stats["corrosion_pitting"].count == 4

    def test_severity_counts(self):
        cls = (
            _make_classifications(2, severity="low") +
            _make_classifications(3, severity="high")
        )
        result = aggregate(cls, _make_seg_masks(5), 10000, 300000, "high", "erosion_hole")
        assert result.severity_counts.get("low", 0) == 2
        assert result.severity_counts.get("high", 0) == 3

    def test_n_requires_review(self):
        cls = _make_classifications(3)
        cls[0]["requires_human_review"] = True
        cls[1]["requires_human_review"] = True
        result = aggregate(cls, _make_seg_masks(3), 5000, 300000, "medium", "erosion_hole")
        assert result.n_requires_review == 2

    def test_erosion_pct_uses_screen_area(self):
        result = aggregate(
            _make_classifications(1), _make_seg_masks(1),
            composite_defect_area_px=30000,
            screen_area_px=300000,
            overall_severity="medium",
            dominant_failure_type="erosion_hole",
        )
        assert result.erosion_pct == pytest.approx(10.0, rel=0.01)

    def test_largest_defect_identified(self):
        masks = [
            {"detection_index": 0, "defect_area_px": 1000, "fill_ratio": 0.5},
            {"detection_index": 1, "defect_area_px": 9000, "fill_ratio": 0.5},
        ]
        cls = _make_classifications(2)
        result = aggregate(cls, masks, 9000, 300000, "medium", "erosion_hole")
        assert result.largest_defect_area_px == 9000


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _make_report(image_id="img_test") -> QuantificationReport:
    from datetime import datetime, timezone
    from src.quantification.aggregator import AggregatedMetrics, FailureTypeStats

    ft_stats = {"erosion_hole": FailureTypeStats(count=2, area_px=8000, area_pct_of_screen=2.67)}
    metrics = AggregatedMetrics(
        n_defects=2, composite_defect_area_px=8000,
        composite_defect_area_pct=2.67, erosion_pct=2.67,
        defect_density_per_10k_px=0.067, n_requires_review=0,
        largest_defect_area_px=5000, largest_defect_failure_type="erosion_hole",
        overall_severity="low", dominant_failure_type="erosion_hole",
        failure_type_stats=ft_stats, severity_counts={"low": 2},
    )
    cls = _make_classifications(2)
    segs = _make_seg_masks(2)
    return build_report(
        image_id=image_id, source_filename="test.jpg",
        image_width=640, image_height=640,
        screen_area_px=300000,
        classifications=cls, seg_masks=segs,
        metrics=metrics, pixels_per_mm=None,
        run_timestamp=datetime.now(timezone.utc).isoformat(),
    )


class TestBuildReport:
    def test_returns_quantification_report(self):
        assert isinstance(_make_report(), QuantificationReport)

    def test_defect_count(self):
        assert _make_report().n_defects == 2

    def test_scale_not_calibrated_when_no_ppm(self):
        r = _make_report()
        assert r.scale_calibrated is False
        assert r.pixels_per_mm is None
        assert all(d.equivalent_diameter_mm is None for d in r.defects)

    def test_mm_computed_when_calibrated(self):
        from datetime import datetime, timezone
        from src.quantification.aggregator import AggregatedMetrics, FailureTypeStats
        ft_stats = {"erosion_hole": FailureTypeStats(count=1, area_px=1000, area_pct_of_screen=0.33)}
        metrics = AggregatedMetrics(
            n_defects=1, composite_defect_area_px=1000,
            composite_defect_area_pct=0.33, erosion_pct=0.33,
            defect_density_per_10k_px=0.033, n_requires_review=0,
            largest_defect_area_px=1000, largest_defect_failure_type="erosion_hole",
            overall_severity="low", dominant_failure_type="erosion_hole",
            failure_type_stats=ft_stats, severity_counts={"low": 1},
        )
        r = build_report(
            "img_cal", "t.jpg", 640, 640, 300000,
            _make_classifications(1), _make_seg_masks(1, 1000),
            metrics, pixels_per_mm=3.5,
            run_timestamp=datetime.now(timezone.utc).isoformat(),
        )
        assert r.scale_calibrated is True
        assert r.defects[0].equivalent_diameter_mm is not None
        assert r.defects[0].equivalent_diameter_mm > 0

    def test_to_dict_has_required_keys(self):
        d = _make_report().to_dict()
        for key in ("image_id", "erosion_pct", "n_defects", "failure_type_breakdown", "defects"):
            assert key in d

    def test_csv_row_flat(self):
        row = _make_report().to_csv_row()
        assert "erosion_pct" in row
        assert "dominant_failure_type" in row
        # No nested dicts in CSV row
        assert all(not isinstance(v, dict) for v in row.values())


class TestExportCsv:
    def test_creates_file(self, tmp_path):
        reports = [_make_report("a"), _make_report("b")]
        out = tmp_path / "out.csv"
        export_csv(reports, out)
        assert out.exists()

    def test_row_count(self, tmp_path):
        import csv
        reports = [_make_report("a"), _make_report("b"), _make_report("c")]
        out = tmp_path / "out.csv"
        export_csv(reports, out)
        with out.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3

    def test_empty_list_no_file(self, tmp_path):
        out = tmp_path / "out.csv"
        export_csv([], out)
        assert not out.exists()


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------

class TestQuantificationStore:
    def test_insert_and_retrieve(self, tmp_path):
        db = tmp_path / "q.db"
        SessionFactory = ensure_table(db)
        r = _make_report()
        with SessionFactory() as session:
            upsert_report(session, r)
            session.commit()
        with SessionFactory() as session:
            records = get_all_reports(session)
        assert len(records) == 1
        assert records[0].image_id == "img_test"

    def test_erosion_persisted(self, tmp_path):
        db = tmp_path / "q.db"
        SessionFactory = ensure_table(db)
        r = _make_report()
        with SessionFactory() as session:
            upsert_report(session, r)
            session.commit()
        with SessionFactory() as session:
            recs = get_all_reports(session)
        assert recs[0].erosion_pct == pytest.approx(2.67, rel=0.01)

    def test_upsert_replaces(self, tmp_path):
        db = tmp_path / "q.db"
        SessionFactory = ensure_table(db)
        r1 = _make_report()
        r1.erosion_pct = 1.0
        r2 = _make_report()
        r2.erosion_pct = 9.9
        with SessionFactory() as session:
            upsert_report(session, r1)
            session.commit()
        with SessionFactory() as session:
            upsert_report(session, r2)
            session.commit()
        with SessionFactory() as session:
            recs = get_all_reports(session)
        assert len(recs) == 1
        assert recs[0].erosion_pct == pytest.approx(9.9, rel=0.01)

    def test_defects_json_roundtrip(self, tmp_path):
        db = tmp_path / "q.db"
        SessionFactory = ensure_table(db)
        with SessionFactory() as session:
            upsert_report(session, _make_report())
            session.commit()
        with SessionFactory() as session:
            recs = get_all_reports(session)
        defects = json.loads(recs[0].defects_json)
        assert len(defects) == 2
        assert "failure_type" in defects[0]
        assert "equivalent_diameter_px" in defects[0]
