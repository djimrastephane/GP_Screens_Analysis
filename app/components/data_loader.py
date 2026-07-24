"""Cached database loading functions for the Streamlit app."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


def get_db_path() -> str:
    """Return absolute path to the SQLite database."""
    return str(Path(__file__).resolve().parents[2] / "data" / "processed" / "images.db")


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@st.cache_data(ttl=300)
def load_summary_df(db_path: str) -> pd.DataFrame:
    """One row per image — all pipeline summary metrics joined."""
    from sqlalchemy import create_engine
    engine = create_engine(f"sqlite:///{db_path}", echo=False)

    with engine.connect() as conn:
        q = pd.read_sql(
            """
            SELECT
                q.image_id,
                q.source_filename,
                q.erosion_pct,
                q.n_defects,
                q.composite_defect_area_px,
                q.screen_region_area_px,
                q.defect_density_per_10k_px,
                q.n_requires_review,
                q.overall_severity,
                q.dominant_failure_type,
                q.scale_calibrated,
                q.pixels_per_mm,
                q.failure_type_breakdown_json,
                q.defects_json,
                q.run_timestamp,
                a.annotated_path,
                a.panel_path,
                p.preprocessed_png_path,
                m.laplacian_variance,
                m.mean_brightness,
                m.brightness_std,
                m.quality_flag,
                m.well_name,
                m.completion_zone,
                m.screen_type,
                m.scale_reference_present,
                m.source_path
            FROM quantification_reports q
            LEFT JOIN annotation_runs a ON q.image_id = a.image_id
            LEFT JOIN preprocessed_images p ON q.image_id = p.image_id
            LEFT JOIN image_metadata m ON q.image_id = m.image_id
            ORDER BY q.source_filename
            """,
            conn,
        )

    # Parse failure type breakdown
    def _dominant_count(row) -> int:
        ft = json.loads(row["failure_type_breakdown_json"] or "{}")
        stats = ft.get(row["dominant_failure_type"], {})
        return stats.get("count", 0)

    q["dominant_count"] = q.apply(_dominant_count, axis=1)

    # Aggregate defect-level stats: mean confidence, largest and average defect size
    def _defect_stats(defects_json: str) -> dict:
        defects = json.loads(defects_json or "[]")
        if not defects:
            return {"mean_confidence": None, "max_defect_area_pct": None,
                    "avg_defect_diameter_px": None, "max_defect_diameter_px": None}
        confidences = [d.get("confidence", 0) for d in defects]
        areas = [d.get("defect_area_pct_of_screen", 0) for d in defects]
        diameters = [d.get("equivalent_diameter_px", 0) for d in defects]
        return {
            "mean_confidence": sum(confidences) / len(confidences),
            "max_defect_area_pct": max(areas),
            "avg_defect_diameter_px": sum(diameters) / len(diameters),
            "max_defect_diameter_px": max(diameters),
        }

    stats = q["defects_json"].apply(_defect_stats).apply(pd.Series)
    q = pd.concat([q.drop(columns=["defects_json"]), stats], axis=1)

    severity_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    q["severity_order"] = q["overall_severity"].map(severity_order).fillna(0)

    return q


@st.cache_data(ttl=300)
def load_defects_df(db_path: str) -> pd.DataFrame:
    """One row per defect — all detections with image context + model reasoning."""
    from sqlalchemy import create_engine
    engine = create_engine(f"sqlite:///{db_path}", echo=False)

    with engine.connect() as conn:
        q_df = pd.read_sql(
            "SELECT image_id, source_filename, defects_json FROM quantification_reports",
            conn,
        )

    rows = []
    for _, row in q_df.iterrows():
        defects = json.loads(row["defects_json"] or "[]")
        for d in defects:
            rows.append({
                "image_id": row["image_id"],
                "source_filename": row["source_filename"],
                **d,
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Join in per-detection reasoning from classification_runs. defects_json has no
    # reasoning field; classifications_json has it keyed by detection_index inside a
    # JSON blob, with no shared SQL join key besides image_id — merge python-side.
    try:
        with engine.connect() as conn:
            cls_df = pd.read_sql(
                "SELECT image_id, classifications_json FROM classification_runs "
                "ORDER BY image_id, run_timestamp",
                conn,
            )
        reasoning_map: dict[tuple[str, int], str] = {}
        for _, r in cls_df.iterrows():
            for c in json.loads(r["classifications_json"] or "[]"):
                # ORDER BY run_timestamp above: latest classification run wins per
                # (image_id, detection_index) if an image was ever reclassified.
                reasoning_map[(r["image_id"], c.get("detection_index", -1))] = c.get("reasoning", "")
        df["reasoning"] = df.apply(
            lambda row: reasoning_map.get((row["image_id"], row["detection_index"]), ""), axis=1,
        )
    except Exception:
        df["reasoning"] = ""

    return df


@st.cache_data(ttl=300)
def load_review_status_df(db_path: str) -> pd.DataFrame:
    """One row per (image_id, detection_index) human review decision."""
    from sqlalchemy import create_engine
    engine = create_engine(f"sqlite:///{db_path}", echo=False)

    try:
        with engine.connect() as conn:
            return pd.read_sql(
                "SELECT id, image_id, detection_index, reviewed, note, reviewed_at "
                "FROM review_status",
                conn,
            )
    except Exception:
        return pd.DataFrame(columns=["id", "image_id", "detection_index", "reviewed", "note", "reviewed_at"])


@st.cache_data(ttl=300)
def load_failure_type_totals(db_path: str) -> dict[str, int]:
    """Aggregate failure type counts across all images."""
    from sqlalchemy import create_engine
    engine = create_engine(f"sqlite:///{db_path}", echo=False)

    with engine.connect() as conn:
        rows = pd.read_sql(
            "SELECT failure_type_breakdown_json FROM quantification_reports", conn
        )

    totals: dict[str, int] = {}
    for _, row in rows.iterrows():
        ft = json.loads(row["failure_type_breakdown_json"] or "{}")
        for ftype, stats in ft.items():
            totals[ftype] = totals.get(ftype, 0) + stats.get("count", 0)
    return totals


@st.cache_data(ttl=300)
def load_pdf_records(db_path: str) -> pd.DataFrame:
    """Load generated PDF report records."""
    from sqlalchemy import create_engine
    engine = create_engine(f"sqlite:///{db_path}", echo=False)

    try:
        with engine.connect() as conn:
            return pd.read_sql(
                "SELECT * FROM report_records ORDER BY report_type, source_filename",
                conn,
            )
    except Exception:
        return pd.DataFrame()
