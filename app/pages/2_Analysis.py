"""Per-image deep dive — annotated overlay, defect table, failure breakdown."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from app.components.data_loader import get_db_path, load_summary_df
from app.components.charts import failure_type_breakdown_pie
from app.components.image_viewer import show_three_panel, show_panel

st.set_page_config(page_title="Analysis", page_icon="🔍", layout="wide")

with st.sidebar:
    st.title("GP Screen Analysis")
    st.divider()
    role = st.radio("View mode", ["Management", "Engineering"], index=1)

db_path = get_db_path()
df = load_summary_df(db_path)

st.title("Per-Image Analysis")

# Image selector
filenames = df["source_filename"].tolist()
selected = st.selectbox("Select image", filenames, index=0)
rec = df[df["source_filename"] == selected].iloc[0]

st.divider()

# ---- Key metrics row ----
k1, k2, k3, k4, k5 = st.columns(5)
sev_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴", "critical": "🔴"}
with k1:
    st.metric("Erosion %", f"{rec['erosion_pct']:.1f}%")
with k2:
    st.metric("Defects", int(rec["n_defects"]))
with k3:
    sev = rec["overall_severity"]
    st.metric("Severity", f"{sev_emoji.get(sev,'')} {sev.capitalize()}")
with k4:
    st.metric("Dominant Type",
              rec["dominant_failure_type"].replace("_", " ").title())
with k5:
    n_rev = int(rec["n_requires_review"])
    st.metric("Need Review", n_rev, delta="⚠" if n_rev else "✓",
              delta_color="inverse" if n_rev else "off")

st.divider()

# ---- Image comparison ----
if role == "Engineering":
    tab_panel, tab_three = st.tabs(["3-Panel Comparison", "Annotated Overlay"])
    with tab_panel:
        show_panel(rec.get("panel_path"))
        st.caption("Left: Original  |  Centre: Annotated with failure types  |  Right: Binary mask")
    with tab_three:
        show_three_panel(
            rec.get("preprocessed_png_path"),
            rec.get("annotated_path"),
            None,
            captions=("Preprocessed", "Annotated", ""),
        )
else:
    if rec.get("annotated_path") and Path(rec["annotated_path"]).exists():
        st.image(str(rec["annotated_path"]), use_container_width=True)
        st.caption(f"Annotated overlay — {selected}")

st.divider()

# ---- Failure type breakdown + chart ----
col_chart, col_table = st.columns([1, 2])

ft_breakdown = json.loads(rec.get("failure_type_breakdown_json") or "{}")
with col_chart:
    st.subheader("Failure Type Breakdown")
    st.plotly_chart(failure_type_breakdown_pie(ft_breakdown),
                    use_container_width=True)

with col_table:
    st.subheader("Breakdown Table")
    if ft_breakdown:
        rows = []
        for ft, stats in ft_breakdown.items():
            rows.append({
                "Failure Type": ft.replace("_", " ").title(),
                "Count": stats["count"],
                "Area (px)": f"{stats['area_px']:,}",
                "Area % screen": f"{stats['area_pct_of_screen']:.2f}%",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

# ---- Per-defect detail (Engineering only) ----
if role == "Engineering":
    st.divider()
    st.subheader("Per-Defect Detail")

    from sqlalchemy import create_engine, text
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT defects_json FROM quantification_reports WHERE image_id = :iid"),
            {"iid": rec["image_id"]},
        )
        row = result.fetchone()

    if row:
        defects = json.loads(row[0] or "[]")
        det_rows = []
        for d in defects:
            det_rows.append({
                "#": d["detection_index"],
                "Failure Type": d["failure_type"].replace("_", " ").title(),
                "Severity": d["severity"].capitalize(),
                "Confidence": round(d["confidence"], 2),
                "Area % screen": round(d["defect_area_pct_of_screen"], 2),
                "Diameter (px)": round(d["equivalent_diameter_px"], 1),
                "Review": "⚠ Yes" if d["requires_human_review"] else "—",
            })

        det_df = pd.DataFrame(det_rows)
        st.dataframe(
            det_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0, max_value=1, format="%.2f"),
            },
        )

    # Notes
    st.divider()
    if not rec.get("scale_calibrated"):
        st.info(
            "**Scale not calibrated.** Diameter measurements are pixel-based only. "
            "Set `pixels_per_mm` in `configs/severity_config.yaml` for physical measurements."
        )
    if n_rev > 0:
        st.warning(
            f"**{n_rev} detection(s) flagged for human review.** "
            "Classifier confidence below threshold — verify before use in decisions."
        )
