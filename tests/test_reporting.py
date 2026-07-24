"""Unit tests for src/reporting modules."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reporting.charts import erosion_bar_chart, failure_type_chart, severity_pie_chart
from src.reporting.store import ensure_table, get_reports, insert_report


# ---------------------------------------------------------------------------
# charts
# ---------------------------------------------------------------------------

class TestErosionBarChart:
    def test_creates_png(self, tmp_path):
        out = tmp_path / "erosion.png"
        erosion_bar_chart(
            filenames=["img_a.jpg", "img_b.jpg"],
            erosion_pcts=[25.0, 40.0],
            severities=["medium", "high"],
            output_path=out,
        )
        assert out.exists() and out.stat().st_size > 0

    def test_single_image(self, tmp_path):
        out = tmp_path / "single.png"
        erosion_bar_chart(["only.jpg"], [15.0], ["low"], out)
        assert out.exists()

    def test_all_severities(self, tmp_path):
        out = tmp_path / "all_sev.png"
        erosion_bar_chart(
            ["a", "b", "c", "d"],
            [3.0, 12.0, 35.0, 65.0],
            ["low", "medium", "high", "critical"],
            out,
        )
        assert out.exists()


class TestFailureTypeChart:
    def test_creates_png(self, tmp_path):
        out = tmp_path / "ft.png"
        failure_type_chart({"erosion_hole": 5, "corrosion_pitting": 12}, out)
        assert out.exists() and out.stat().st_size > 0

    def test_empty_dict_no_crash(self, tmp_path):
        out = tmp_path / "empty.png"
        failure_type_chart({}, out)
        # Empty dict returns without writing — should not raise

    def test_all_known_types(self, tmp_path):
        out = tmp_path / "all_types.png"
        counts = {
            "erosion_hole": 10, "corrosion_pitting": 8,
            "wire_wrap_failure": 3, "plugging_partial": 2,
        }
        failure_type_chart(counts, out)
        assert out.exists()


class TestSeverityPieChart:
    def test_creates_png(self, tmp_path):
        out = tmp_path / "sev.png"
        severity_pie_chart({"low": 3, "medium": 4, "high": 2}, out)
        assert out.exists() and out.stat().st_size > 0

    def test_empty_dict_no_crash(self, tmp_path):
        out = tmp_path / "empty_sev.png"
        severity_pie_chart({}, out)

    def test_single_severity(self, tmp_path):
        out = tmp_path / "single_sev.png"
        severity_pie_chart({"medium": 9}, out)
        assert out.exists()


# ---------------------------------------------------------------------------
# per_image (PDF generation)
# ---------------------------------------------------------------------------

class TestPerImageReport:
    def _make_quant_record(self):
        """Minimal mock of a QuantificationRecord."""
        import json
        from types import SimpleNamespace

        ft_breakdown = {
            "erosion_hole": {"count": 3, "area_px": 5000, "area_pct_of_screen": 1.62},
            "corrosion_pitting": {"count": 5, "area_px": 9000, "area_pct_of_screen": 2.92},
        }
        defects = [
            {
                "detection_index": i,
                "failure_type": "erosion_hole" if i % 2 == 0 else "corrosion_pitting",
                "severity": "medium",
                "confidence": 0.75,
                "requires_human_review": i == 0,
                "defect_area_px": 800,
                "defect_area_pct_of_screen": 0.26,
                "equivalent_diameter_px": 31.9,
                "fill_ratio": 0.78,
                "equivalent_diameter_mm": None,
            }
            for i in range(3)
        ]
        return SimpleNamespace(
            image_id="test_img",
            source_filename="test_screen.jpg",
            image_width=640, image_height=640,
            screen_region_area_px=308480,
            scale_calibrated=False, pixels_per_mm=None,
            n_defects=8,
            composite_defect_area_px=14000,
            erosion_pct=4.54,
            defect_density_per_10k_px=0.259,
            n_requires_review=1,
            overall_severity="medium",
            dominant_failure_type="corrosion_pitting",
            failure_type_breakdown_json=json.dumps(ft_breakdown),
            severity_breakdown_json=json.dumps({"low": 2, "medium": 6}),
            defects_json=json.dumps(defects),
            run_timestamp="2026-05-21T00:00:00Z",
        )

    def test_creates_pdf(self, tmp_path):
        from src.reporting.per_image import build_per_image_report
        rec = self._make_quant_record()
        out = tmp_path / "test_report.pdf"
        result = build_per_image_report(rec, None, out)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_pdf_has_pdf_header(self, tmp_path):
        from src.reporting.per_image import build_per_image_report
        rec = self._make_quant_record()
        out = tmp_path / "test.pdf"
        build_per_image_report(rec, None, out)
        with out.open("rb") as f:
            header = f.read(4)
        assert header == b"%PDF"

    def test_no_annotation_no_crash(self, tmp_path):
        from src.reporting.per_image import build_per_image_report
        rec = self._make_quant_record()
        out = tmp_path / "no_ann.pdf"
        build_per_image_report(rec, None, out)
        assert out.exists()


# ---------------------------------------------------------------------------
# summary (PDF generation)
# ---------------------------------------------------------------------------

class TestSummaryReport:
    def _make_records(self, n=3):
        import json
        from types import SimpleNamespace

        records = []
        for i in range(n):
            ft = {"erosion_hole": {"count": 2, "area_px": 3000, "area_pct_of_screen": 0.97},
                  "corrosion_pitting": {"count": 4, "area_px": 6000, "area_pct_of_screen": 1.95}}
            records.append(SimpleNamespace(
                image_id=f"img_{i}",
                source_filename=f"screen_{i}.jpg",
                image_width=640, image_height=640,
                screen_region_area_px=308480,
                scale_calibrated=False, pixels_per_mm=None,
                n_defects=6,
                composite_defect_area_px=9000,
                erosion_pct=10.0 + i * 8.0,
                defect_density_per_10k_px=0.19,
                n_requires_review=i % 2,
                overall_severity=["low", "medium", "high"][i % 3],
                dominant_failure_type="corrosion_pitting",
                failure_type_breakdown_json=json.dumps(ft),
                severity_breakdown_json=json.dumps({"medium": 6}),
                defects_json=json.dumps([]),
                run_timestamp="2026-05-21T00:00:00Z",
            ))
        return records

    def test_creates_pdf(self, tmp_path):
        from src.reporting.summary import build_summary_report
        recs = self._make_records(3)
        out = tmp_path / "summary.pdf"
        result = build_summary_report(recs, {}, tmp_path / "charts", out)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_pdf_header(self, tmp_path):
        from src.reporting.summary import build_summary_report
        recs = self._make_records(2)
        out = tmp_path / "sum.pdf"
        build_summary_report(recs, {}, tmp_path / "charts", out)
        with out.open("rb") as f:
            assert f.read(4) == b"%PDF"

    def test_single_image_no_crash(self, tmp_path):
        from src.reporting.summary import build_summary_report
        recs = self._make_records(1)
        out = tmp_path / "single.pdf"
        build_summary_report(recs, {}, tmp_path / "charts", out)
        assert out.exists()

    def test_charts_created(self, tmp_path):
        from src.reporting.summary import build_summary_report
        recs = self._make_records(4)
        charts_dir = tmp_path / "charts"
        build_summary_report(recs, {}, charts_dir, tmp_path / "s.pdf")
        assert (charts_dir / "erosion_bar.png").exists()
        assert (charts_dir / "failure_type_bar.png").exists()
        assert (charts_dir / "severity_pie.png").exists()


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------

class TestReportStore:
    def test_insert_and_retrieve(self, tmp_path):
        db = tmp_path / "rep.db"
        SessionFactory = ensure_table(db)
        with SessionFactory() as session:
            insert_report(session, "per_image", "/out/a.pdf", "2026-05-21T00:00:00Z",
                          image_id="img_a", source_filename="a.jpg")
            session.commit()
        with SessionFactory() as session:
            recs = get_reports(session)
        assert len(recs) == 1
        assert recs[0].report_type == "per_image"

    def test_filter_by_type(self, tmp_path):
        db = tmp_path / "rep2.db"
        SessionFactory = ensure_table(db)
        with SessionFactory() as session:
            insert_report(session, "per_image", "/a.pdf", "ts", image_id="img_a")
            insert_report(session, "per_image", "/b.pdf", "ts", image_id="img_b")
            insert_report(session, "summary", "/s.pdf", "ts")
            session.commit()
        with SessionFactory() as session:
            per = get_reports(session, "per_image")
            summ = get_reports(session, "summary")
        assert len(per) == 2
        assert len(summ) == 1

    def test_rerun_replaces_existing_record_instead_of_duplicating(self, tmp_path):
        db = tmp_path / "rerun.db"
        SessionFactory = ensure_table(db)
        with SessionFactory() as session:
            insert_report(session, "per_image", "/out/a_v1.pdf", "ts1", image_id="img_a")
            session.commit()
        with SessionFactory() as session:
            insert_report(session, "per_image", "/out/a_v2.pdf", "ts2", image_id="img_a")
            session.commit()
        with SessionFactory() as session:
            recs = get_reports(session, "per_image")
        assert len(recs) == 1
        assert recs[0].pdf_path == "/out/a_v2.pdf"
        assert recs[0].run_timestamp == "ts2"

    def test_rerun_replaces_summary_record_instead_of_duplicating(self, tmp_path):
        db = tmp_path / "rerun_summary.db"
        SessionFactory = ensure_table(db)
        with SessionFactory() as session:
            insert_report(session, "summary", "/out/s_v1.pdf", "ts1")
            session.commit()
        with SessionFactory() as session:
            insert_report(session, "summary", "/out/s_v2.pdf", "ts2")
            session.commit()
        with SessionFactory() as session:
            recs = get_reports(session, "summary")
        assert len(recs) == 1
        assert recs[0].pdf_path == "/out/s_v2.pdf"

    def test_multiple_per_image_reports(self, tmp_path):
        db = tmp_path / "multi.db"
        SessionFactory = ensure_table(db)
        with SessionFactory() as session:
            for i in range(5):
                insert_report(session, "per_image", f"/img_{i}.pdf", "ts",
                              image_id=f"img_{i}", source_filename=f"img_{i}.jpg")
            session.commit()
        with SessionFactory() as session:
            recs = get_reports(session, "per_image")
        assert len(recs) == 5
