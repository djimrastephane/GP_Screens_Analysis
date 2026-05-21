"""Home page — campaign overview KPIs and summary charts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from app.components.data_loader import (
    get_db_path,
    load_failure_type_totals,
    load_summary_df,
)
from app.components.charts import erosion_bar, failure_type_bar, severity_pie
from app.components.interpretation import EROSION_PCT_DEFINITION

st.set_page_config(
    page_title="GP Screen Failure Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.title("GP Screen Analysis")
    st.caption("Gravel Pack Failure Investigation")
    st.divider()
    role = st.radio(
        "View mode",
        ["Management", "Engineering"],
        index=1,
    )

db_path = get_db_path()
try:
    df = load_summary_df(db_path)
    ft_totals = load_failure_type_totals(db_path)
except Exception as e:
    st.error(f"Database not found. Run the batch scripts first.\n\n{e}")
    st.stop()

st.title("GP Screen Failure Analysis")
st.caption("Gravel pack screen inspection · automated computer vision pipeline")
st.divider()

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Images analysed", len(df))
with k2:
    st.metric("Mean erosion %", f"{df['erosion_pct'].mean():.1f}%",
              help=EROSION_PCT_DEFINITION)
with k3:
    max_row = df.loc[df["erosion_pct"].idxmax()]
    st.metric("Max erosion %", f"{max_row['erosion_pct']:.1f}%",
              delta=max_row["source_filename"], delta_color="off",
              help=EROSION_PCT_DEFINITION)
with k4:
    st.metric("Total defects", int(df["n_defects"].sum()))
with k5:
    n_rev = int(df["n_requires_review"].sum())
    st.metric("Need review", n_rev,
              delta="detections flagged" if n_rev else "none flagged",
              delta_color="inverse" if n_rev else "off")

st.divider()

if role == "Management":
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.subheader("Erosion % by Image")
        st.plotly_chart(erosion_bar(df), use_container_width=True)
    with col_r:
        st.subheader("Severity Distribution")
        st.plotly_chart(severity_pie(df["overall_severity"].value_counts().to_dict()),
                        use_container_width=True)
else:
    col_l, col_m, col_r = st.columns([2.5, 1.8, 1])
    with col_l:
        st.subheader("Erosion % by Image")
        st.plotly_chart(erosion_bar(df), use_container_width=True)
    with col_m:
        st.subheader("Failure Type Distribution")
        st.plotly_chart(failure_type_bar(ft_totals), use_container_width=True)
    with col_r:
        st.subheader("Severity")
        st.plotly_chart(severity_pie(df["overall_severity"].value_counts().to_dict()),
                        use_container_width=True)

st.divider()
st.subheader("Image Summary")

display_df = df[["source_filename", "erosion_pct", "n_defects",
                  "overall_severity", "dominant_failure_type", "n_requires_review"]].copy()
display_df.columns = ["Filename", "Erosion %", "Defects", "Severity",
                       "Dominant Failure", "Review"]
display_df["Erosion %"] = display_df["Erosion %"].round(1)
display_df["Dominant Failure"] = display_df["Dominant Failure"].str.replace("_", " ").str.title()

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Erosion %": st.column_config.ProgressColumn(
            "Erosion %", min_value=0, max_value=100, format="%.1f%%"),
        "Review": st.column_config.NumberColumn("Review ⚠", format="%d"),
    },
)
