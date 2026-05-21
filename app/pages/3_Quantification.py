"""Interactive quantification charts and data export."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from app.components.data_loader import (
    get_db_path,
    load_defects_df,
    load_failure_type_totals,
    load_summary_df,
)
from app.components.charts import (
    erosion_bar,
    failure_type_bar,
    scatter_erosion_defects,
    severity_pie,
)
from app.components.interpretation import EROSION_PCT_DEFINITION

st.set_page_config(page_title="Quantification", page_icon="📊", layout="wide")

with st.sidebar:
    st.title("GP Screen Analysis")
    st.divider()
    role = st.radio("View mode", ["Management", "Engineering"], index=1)

db_path = get_db_path()
df = load_summary_df(db_path)
ft_totals = load_failure_type_totals(db_path)

st.title("Quantification")
st.caption("Erosion measurements, defect metrics, and failure type analysis")
st.divider()

# ---- Row 1: erosion bar + severity pie ----
col_l, col_r = st.columns([3, 1])
with col_l:
    st.subheader("Erosion %  by Image")
    st.caption(
        "Dashed lines: 20 % (High threshold) and 50 % (Critical threshold). "
        "Definition: " + EROSION_PCT_DEFINITION
    )
    st.plotly_chart(erosion_bar(df), use_container_width=True)
with col_r:
    st.subheader("Severity")
    sev_counts = df["overall_severity"].value_counts().to_dict()
    st.plotly_chart(severity_pie(sev_counts), use_container_width=True)

st.divider()

# ---- Row 2: failure type bar + scatter ----
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Failure Type Distribution")
    st.plotly_chart(failure_type_bar(ft_totals), use_container_width=True)
with col_b:
    st.subheader("Erosion % vs Defect Count")
    st.plotly_chart(scatter_erosion_defects(df), use_container_width=True)

# ---- Engineering-only: full metrics table + export ----
if role == "Engineering":
    st.divider()
    st.subheader("Full Metrics Table")

    cols_wanted = [
        "source_filename", "erosion_pct", "n_defects",
        "max_defect_area_pct", "avg_defect_diameter_px",
        "defect_density_per_10k_px", "mean_confidence",
        "overall_severity", "dominant_failure_type", "n_requires_review",
    ]
    cols_wanted = [c for c in cols_wanted if c in df.columns]
    display = df[cols_wanted].copy()
    col_labels = {
        "source_filename": "Filename",
        "erosion_pct": "Erosion %",
        "n_defects": "Defects",
        "max_defect_area_pct": "Largest Defect %",
        "avg_defect_diameter_px": "Avg Diameter (px)",
        "defect_density_per_10k_px": "Density / 10k px",
        "mean_confidence": "Mean Confidence",
        "overall_severity": "Severity",
        "dominant_failure_type": "Dominant Failure",
        "n_requires_review": "Review",
    }
    display.rename(columns=col_labels, inplace=True)
    display["Erosion %"] = display["Erosion %"].round(2)
    if "Density / 10k px" in display.columns:
        display["Density / 10k px"] = display["Density / 10k px"].round(4)
    if "Largest Defect %" in display.columns:
        display["Largest Defect %"] = display["Largest Defect %"].round(2)
    if "Avg Diameter (px)" in display.columns:
        display["Avg Diameter (px)"] = display["Avg Diameter (px)"].round(1)
    if "Mean Confidence" in display.columns:
        display["Mean Confidence"] = display["Mean Confidence"].round(3)
    if "Dominant Failure" in display.columns:
        display["Dominant Failure"] = display["Dominant Failure"].str.replace("_", " ").str.title()

    st.dataframe(
        display,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Erosion %": st.column_config.ProgressColumn(
                "Erosion %", min_value=0, max_value=100, format="%.2f%%"),
            "Mean Confidence": st.column_config.ProgressColumn(
                "Mean Confidence", min_value=0, max_value=1, format="%.0%%"),
        },
    )

    # CSV export
    csv = display.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download metrics CSV",
        data=csv,
        file_name="gp_screen_quantification.csv",
        mime="text/csv",
    )

    # Per-defect table
    st.divider()
    st.subheader("All Defects")
    defects_df = load_defects_df(db_path)
    if not defects_df.empty:
        show_cols = [
            "source_filename", "failure_type", "severity",
            "confidence", "defect_area_pct_of_screen",
            "equivalent_diameter_px", "requires_human_review",
        ]
        show_cols = [c for c in show_cols if c in defects_df.columns]
        display_dets = defects_df[show_cols].copy()
        display_dets.columns = [c.replace("_", " ").title() for c in show_cols]

        st.dataframe(display_dets, hide_index=True, use_container_width=True,
                     height=300)

        det_csv = defects_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download defects CSV",
            data=det_csv,
            file_name="gp_screen_all_defects.csv",
            mime="text/csv",
        )
