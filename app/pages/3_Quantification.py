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
    st.subheader("Erosion % by Image")
    st.caption("Dashed lines: 20% (medium threshold) and 50% (high threshold)")
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

    display = df[[
        "source_filename", "erosion_pct", "n_defects",
        "composite_defect_area_px", "screen_region_area_px",
        "defect_density_per_10k_px", "overall_severity",
        "dominant_failure_type", "n_requires_review",
    ]].copy()
    display.columns = [
        "Filename", "Erosion %", "Defects",
        "Defect Area (px)", "Screen Area (px)",
        "Density / 10k px", "Severity",
        "Dominant Failure", "Review",
    ]
    display["Erosion %"] = display["Erosion %"].round(2)
    display["Density / 10k px"] = display["Density / 10k px"].round(4)
    display["Dominant Failure"] = display["Dominant Failure"].str.replace("_", " ").str.title()

    st.dataframe(display, hide_index=True, use_container_width=True)

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
