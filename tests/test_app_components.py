"""Unit tests for app/components (no running Streamlit required)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# charts
# ---------------------------------------------------------------------------

class TestCharts:
    def _sample_df(self, n=3) -> pd.DataFrame:
        return pd.DataFrame({
            "source_filename": [f"img_{i}.jpg" for i in range(n)],
            "erosion_pct": [10.0 + i * 8 for i in range(n)],
            "n_defects": [5 + i for i in range(n)],
            "overall_severity": ["low", "medium", "high"][:n],
            "dominant_failure_type": ["erosion_hole"] * n,
            "n_requires_review": [0, 1, 0][:n],
        })

    def test_erosion_bar_returns_figure(self):
        import plotly.graph_objects as go
        from app.components.charts import erosion_bar
        fig = erosion_bar(self._sample_df())
        assert isinstance(fig, go.Figure)

    def test_failure_type_bar_returns_figure(self):
        import plotly.graph_objects as go
        from app.components.charts import failure_type_bar
        fig = failure_type_bar({"erosion_hole": 5, "corrosion_pitting": 8})
        assert isinstance(fig, go.Figure)

    def test_failure_type_bar_empty(self):
        import plotly.graph_objects as go
        from app.components.charts import failure_type_bar
        fig = failure_type_bar({})
        assert isinstance(fig, go.Figure)

    def test_severity_pie_returns_figure(self):
        import plotly.graph_objects as go
        from app.components.charts import severity_pie
        fig = severity_pie({"low": 3, "medium": 4})
        assert isinstance(fig, go.Figure)

    def test_scatter_returns_figure(self):
        import plotly.graph_objects as go
        from app.components.charts import scatter_erosion_defects
        fig = scatter_erosion_defects(self._sample_df())
        assert isinstance(fig, go.Figure)

    def test_breakdown_pie_returns_figure(self):
        import plotly.graph_objects as go
        from app.components.charts import failure_type_breakdown_pie
        breakdown = {
            "erosion_hole": {"count": 3},
            "corrosion_pitting": {"count": 5},
        }
        fig = failure_type_breakdown_pie(breakdown)
        assert isinstance(fig, go.Figure)

    def test_erosion_bar_single_row(self):
        from app.components.charts import erosion_bar
        df = self._sample_df(1)
        fig = erosion_bar(df)
        assert fig is not None

    def test_all_severity_colours_covered(self):
        from app.components.charts import _SEVERITY_COLOURS
        for sev in ["low", "medium", "high", "critical"]:
            assert sev in _SEVERITY_COLOURS

    def test_all_failure_type_colours_covered(self):
        from app.components.charts import _FAILURE_TYPE_COLOURS
        for ft in [
            "erosion_hole", "corrosion_pitting", "wire_wrap_failure",
            "screen_collapse", "mechanical_damage",
            "plugging_partial", "plugging_complete", "unknown",
        ]:
            assert ft in _FAILURE_TYPE_COLOURS


# ---------------------------------------------------------------------------
# image_viewer
# ---------------------------------------------------------------------------

class TestSeverityBadge:
    def test_returns_html_string(self):
        from app.components.image_viewer import severity_badge
        badge = severity_badge("medium")
        assert isinstance(badge, str)
        assert "medium" in badge.lower() or "MEDIUM" in badge

    def test_all_severities(self):
        from app.components.image_viewer import severity_badge
        for sev in ["low", "medium", "high", "critical"]:
            result = severity_badge(sev)
            assert len(result) > 0

    def test_unknown_severity_no_crash(self):
        from app.components.image_viewer import severity_badge
        result = severity_badge("super_critical")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# data_loader (non-Streamlit parts)
# ---------------------------------------------------------------------------

class TestDataLoader:
    def test_get_db_path_is_string(self):
        from app.components.data_loader import get_db_path
        path = get_db_path()
        assert isinstance(path, str)
        assert path.endswith("images.db")

    def test_get_project_root_exists(self):
        from app.components.data_loader import get_project_root
        root = get_project_root()
        assert root.exists()
        assert (root / "src").exists()

    def test_db_path_points_to_existing_file(self):
        from app.components.data_loader import get_db_path
        path = get_db_path()
        assert Path(path).exists(), f"DB not found at {path} — run pipeline scripts first"


# ---------------------------------------------------------------------------
# data_loader integration (reads real DB)
# ---------------------------------------------------------------------------

class TestDataLoaderIntegration:
    """These tests require the pipeline to have been run."""

    def _db(self):
        from app.components.data_loader import get_db_path
        return get_db_path()

    def test_summary_df_columns(self):
        from app.components.data_loader import load_summary_df
        df = load_summary_df(self._db())
        for col in ["image_id", "source_filename", "erosion_pct",
                    "n_defects", "overall_severity"]:
            assert col in df.columns

    def test_summary_df_has_rows(self):
        from app.components.data_loader import load_summary_df
        df = load_summary_df(self._db())
        assert len(df) > 0

    def test_erosion_pct_in_valid_range(self):
        from app.components.data_loader import load_summary_df
        df = load_summary_df(self._db())
        assert (df["erosion_pct"] >= 0).all()
        assert (df["erosion_pct"] <= 100).all()

    def test_failure_type_totals_dict(self):
        from app.components.data_loader import load_failure_type_totals
        totals = load_failure_type_totals(self._db())
        assert isinstance(totals, dict)
        assert len(totals) > 0
        assert all(isinstance(v, int) for v in totals.values())

    def test_defects_df_columns(self):
        from app.components.data_loader import load_defects_df
        df = load_defects_df(self._db())
        assert len(df) > 0
        assert "failure_type" in df.columns
        assert "severity" in df.columns

    def test_pdf_records_exist(self):
        from app.components.data_loader import load_pdf_records
        df = load_pdf_records(self._db())
        assert len(df) > 0
        assert "report_type" in df.columns
