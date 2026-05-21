"""Image gallery — browse annotated overlays with key metrics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.components.data_loader import get_db_path, load_summary_df
from app.components.image_viewer import severity_badge

st.set_page_config(page_title="Gallery", page_icon="🖼️", layout="wide")

with st.sidebar:
    st.title("GP Screen Analysis")
    st.divider()
    role = st.radio("View mode", ["Management", "Engineering"], index=1)
    st.divider()
    severity_filter = st.multiselect(
        "Filter by severity",
        ["low", "medium", "high", "critical"],
        default=[],
        placeholder="All severities",
    )
    failure_filter = st.multiselect(
        "Filter by dominant failure",
        ["corrosion_pitting", "erosion_hole", "wire_wrap_failure",
         "screen_collapse", "plugging_partial", "mechanical_damage"],
        default=[],
        placeholder="All failure types",
    )

db_path = get_db_path()
df = load_summary_df(db_path)

if severity_filter:
    df = df[df["overall_severity"].isin(severity_filter)]
if failure_filter:
    df = df[df["dominant_failure_type"].isin(failure_filter)]

st.title("Image Gallery")
st.caption(f"Showing {len(df)} image{'s' if len(df) != 1 else ''}")

if df.empty:
    st.info("No images match the current filters.")
    st.stop()

# Sort options
sort_col, _ = st.columns([2, 5])
with sort_col:
    sort_by = st.selectbox(
        "Sort by",
        ["Erosion % (high→low)", "Erosion % (low→high)", "Filename"],
        index=0,
    )

if sort_by == "Erosion % (high→low)":
    df = df.sort_values("erosion_pct", ascending=False)
elif sort_by == "Erosion % (low→high)":
    df = df.sort_values("erosion_pct", ascending=True)
else:
    df = df.sort_values("source_filename")

cols_per_row = 2 if role == "Management" else 3

for row_start in range(0, len(df), cols_per_row):
    row_df = df.iloc[row_start: row_start + cols_per_row]
    cols = st.columns(cols_per_row)

    for col, (_, rec) in zip(cols, row_df.iterrows()):
        with col:
            # Annotated thumbnail
            ann_path = rec.get("annotated_path")
            if ann_path and Path(ann_path).exists():
                st.image(str(ann_path), use_container_width=True)
            else:
                st.info("Image not available")

            # Caption metrics
            sev = rec["overall_severity"]
            sev_colours = {"low": "green", "medium": "orange",
                           "high": "red", "critical": "red"}
            sev_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴", "critical": "🔴"}

            st.markdown(f"**{rec['source_filename']}**")

            m1, m2, m3 = st.columns(3)
            m1.metric("Erosion", f"{rec['erosion_pct']:.1f}%")
            m2.metric("Defects", int(rec["n_defects"]))
            m3.metric("Severity", f"{sev_emoji.get(sev,'')} {sev.capitalize()}")

            if role == "Engineering":
                dom = rec["dominant_failure_type"].replace("_", " ").title()
                st.caption(f"Dominant: {dom}")
                if rec["n_requires_review"] > 0:
                    st.warning(
                        f"⚠ {int(rec['n_requires_review'])} detection(s) need review",
                        icon=None,
                    )

    st.divider()
