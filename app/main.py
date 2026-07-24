"""Overview — campaign KPIs, engineering assessment, and full data export."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from app.components.nav import render_sidebar_header
from app.components.data_loader import (
    get_db_path,
    load_defects_df,
    load_failure_type_totals,
    load_summary_df,
)
from app.components.charts import (
    erosion_bar,
    failure_type_bar,
    severity_pie,
    scatter_erosion_defects,
)
from app.components.interpretation import (
    EROSION_PCT_DEFINITION,
    campaign_assessment,
    classification_evidence,
    _SEVERITY_BASIS as SEVERITY_BASIS,
)

st.set_page_config(
    page_title="GP Screen Failure Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

role = render_sidebar_header()

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

tab_dash, tab_assess, tab_full = st.tabs(["Dashboard", "Assessment", "Full Data & Export"])

# ============================================================
# Dashboard
# ============================================================
with tab_dash:
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
        if n_rev > 0:
            st.page_link("pages/1_Review_Queue.py", label="Review flagged detections →", icon="🔎")

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

# ============================================================
# Assessment
# ============================================================
with tab_assess:
    assess = campaign_assessment(df)

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

    sev = assess["sev_counts"]
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        st.metric("Screens inspected", assess["n_images"])
    with a2:
        st.metric("Mean erosion %", f"{assess['mean_erosion']:.1f}%")
    with a3:
        st.metric("Max erosion %", f"{assess['max_erosion']:.1f}%")
    with a4:
        n_high = sev.get("high", 0) + sev.get("critical", 0)
        st.metric("High / Critical severity", n_high,
                  help="Number of screens rated High or Critical severity.")

    st.divider()

    st.subheader("Observed Conditions")
    for line in assess["observed_conditions"]:
        st.markdown(f"- {line}")

    st.divider()

    st.subheader("Classification Basis")
    st.caption(
        "For each failure type detected in this campaign, the following summarises the "
        "morphological features that support the classification."
    )
    for ft in assess["failure_types_present"]:
        with st.expander(ft.replace("_", " ").title()):
            st.markdown(classification_evidence(ft))

    st.divider()

    st.subheader("Potential Root Causes")
    st.caption(
        "Derived from the combination of failure types detected across all inspected screens. "
        "These are hypotheses requiring confirmation from production history, fluid chemistry, "
        "and operational records."
    )
    for cause in assess["potential_causes"]:
        st.markdown(f"- {cause}")

    st.divider()

    st.subheader("Severity Distribution")
    sev_order = ["critical", "high", "medium", "low"]
    sev_labels = {"critical": "Critical", "high": "High", "medium": "Medium", "low": "Low"}

    sev_cols = st.columns(4)
    for col, level in zip(sev_cols, sev_order):
        count = sev.get(level, 0)
        col.metric(sev_labels[level], count, help=f"Screens rated {sev_labels[level]} severity.")

    with st.expander("Severity threshold basis"):
        st.caption(SEVERITY_BASIS)

    st.divider()

    st.subheader("Recommended Actions")
    st.caption("Priority order based on detected failure types and severity, most critical first.")
    for i, action in enumerate(assess["recommended_actions"], 1):
        st.markdown(f"{i}. {action}")

# ============================================================
# Full Data & Export
# ============================================================
with tab_full:
    st.subheader("Erosion % vs Defect Count")
    st.plotly_chart(scatter_erosion_defects(df), use_container_width=True)

    st.divider()
    st.subheader("Full Metrics Table")

    cols_wanted = [
        "source_filename", "erosion_pct", "n_defects",
        "max_defect_area_pct", "avg_defect_diameter_px",
        "defect_density_per_10k_px", "mean_confidence",
        "overall_severity", "dominant_failure_type", "n_requires_review",
        "quality_flag",
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
        "quality_flag": "Image Quality",
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
    if "Image Quality" in display.columns:
        display["Image Quality"] = display["Image Quality"].fillna("unknown").str.replace("_", " ").str.title()
    display = display.sort_values("Erosion %", ascending=False)

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

    csv = display.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download metrics CSV",
        data=csv,
        file_name="gp_screen_quantification.csv",
        mime="text/csv",
    )

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

        st.dataframe(display_dets, hide_index=True, use_container_width=True, height=300)

        det_csv = defects_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download defects CSV",
            data=det_csv,
            file_name="gp_screen_all_defects.csv",
            mime="text/csv",
        )
