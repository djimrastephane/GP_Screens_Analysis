"""Campaign-level Engineering Assessment — synthesised from all inspected screens."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.components.data_loader import get_db_path, load_summary_df
from app.components.interpretation import (
    campaign_assessment,
    classification_evidence,
    _SEVERITY_BASIS as SEVERITY_BASIS,
)

st.set_page_config(page_title="Engineering Assessment", page_icon="📋", layout="wide")

with st.sidebar:
    st.title("GP Screen Analysis")
    st.divider()
    st.radio("View mode", ["Management", "Engineering"], index=1, key="role_assess")

db_path = get_db_path()
df = load_summary_df(db_path)

st.title("Engineering Assessment")
st.caption("Campaign-level synthesis of observed conditions, root cause hypotheses, and recommended actions.")
st.divider()

assess = campaign_assessment(df)

# ---- Overall risk banner ----
risk_colours = {"LOW": "green", "MODERATE": "orange", "HIGH": "red", "CRITICAL": "darkred"}
risk = assess["risk"]
colour = risk_colours.get(risk, "grey")

st.markdown(
    f"<div style='background:#f8f8f8;border-left:6px solid {colour};"
    f"padding:12px 20px;border-radius:4px;margin-bottom:16px'>"
    f"<span style='font-size:1.1em;font-weight:700;color:{colour}'>Campaign Risk: {risk}</span>"
    f"<br><span style='color:#555'>{assess['risk_description']}</span>"
    f"</div>",
    unsafe_allow_html=True,
)

# ---- Summary KPIs ----
k1, k2, k3, k4 = st.columns(4)
sev = assess["sev_counts"]
with k1:
    st.metric("Screens inspected", assess["n_images"])
with k2:
    st.metric("Mean erosion %", f"{assess['mean_erosion']:.1f}%")
with k3:
    st.metric("Max erosion %", f"{assess['max_erosion']:.1f}%")
with k4:
    n_high = sev.get("high", 0) + sev.get("critical", 0)
    st.metric("High / Critical severity", n_high,
              help="Number of screens rated High or Critical severity.")

st.divider()

# ---- Observed conditions ----
st.subheader("Observed Conditions")
for line in assess["observed_conditions"]:
    st.markdown(f"- {line}")

st.divider()

# ---- Classification basis ----
st.subheader("Classification Basis")
st.caption(
    "For each failure type detected in this campaign, the following summarises the "
    "morphological features that support the classification."
)
for ft in assess["failure_types_present"]:
    with st.expander(ft.replace("_", " ").title()):
        st.markdown(classification_evidence(ft))

st.divider()

# ---- Potential causes ----
st.subheader("Potential Root Causes")
st.caption(
    "Derived from the combination of failure types detected across all inspected screens. "
    "These are hypotheses requiring confirmation from production history, fluid chemistry, "
    "and operational records."
)
for cause in assess["potential_causes"]:
    st.markdown(f"- {cause}")

st.divider()

# ---- Risk breakdown ----
st.subheader("Severity Distribution")
sev_order = ["critical", "high", "medium", "low"]
sev_labels = {"critical": "Critical", "high": "High", "medium": "Medium", "low": "Low"}
sev_colours_css = {"critical": "darkred", "high": "red", "medium": "orange", "low": "green"}

cols = st.columns(4)
for col, level in zip(cols, sev_order):
    count = sev.get(level, 0)
    col.metric(sev_labels[level], count,
               help=f"Screens rated {sev_labels[level]} severity.")

with st.expander("Severity threshold basis"):
    st.caption(SEVERITY_BASIS)

st.divider()

# ---- Recommended actions ----
st.subheader("Recommended Actions")
st.caption("Priority order based on detected failure types and severity, most critical first.")
for i, action in enumerate(assess["recommended_actions"], 1):
    st.markdown(f"{i}. {action}")

st.divider()

# ---- Per-image summary table ----
st.subheader("Per-Screen Summary")

import pandas as pd

display = df[[
    "source_filename", "erosion_pct", "n_defects",
    "max_defect_area_pct", "mean_confidence",
    "overall_severity", "dominant_failure_type", "n_requires_review",
    "quality_flag",
]].copy()

display.columns = [
    "Filename", "Erosion %", "Defects",
    "Largest Defect %", "Mean Confidence",
    "Severity", "Dominant Failure", "Review",
    "Image Quality",
]
display["Erosion %"] = display["Erosion %"].round(1)
display["Largest Defect %"] = display["Largest Defect %"].round(2)
display["Mean Confidence"] = display["Mean Confidence"].round(3)
display["Dominant Failure"] = display["Dominant Failure"].str.replace("_", " ").str.title()
display["Image Quality"] = display["Image Quality"].fillna("unknown").str.replace("_", " ").str.title()
display = display.sort_values("Erosion %", ascending=False)

st.dataframe(
    display,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Erosion %": st.column_config.ProgressColumn(
            "Erosion %", min_value=0, max_value=100, format="%.1f%%"),
        "Mean Confidence": st.column_config.ProgressColumn(
            "Mean Confidence", min_value=0, max_value=1, format="%.0%%"),
        "Largest Defect %": st.column_config.NumberColumn(
            "Largest Defect %", format="%.2f%%"),
    },
)
