"""Per-image deep dive — annotated overlay, defect table, failure breakdown."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

from app.components.nav import render_sidebar_header
from app.components.data_loader import get_db_path, load_summary_df, load_review_status_df
from app.components.charts import failure_type_breakdown_pie
from app.components.image_viewer import show_three_panel, show_panel
from app.components.interpretation import (
    engineering_interpretation,
    classification_evidence,
    EROSION_PCT_DEFINITION,
    _SEVERITY_BASIS as SEVERITY_BASIS,
)
from src.classification.store import ensure_table as ensure_classification_tables, set_reviewed

st.set_page_config(page_title="Analysis", page_icon="🔍", layout="wide")

role = render_sidebar_header()

db_path = get_db_path()
ensure_classification_tables(db_path)
df = load_summary_df(db_path)

st.title("Per-Image Analysis")

filenames = df["source_filename"].tolist()
preselect = st.query_params.get("image")
default_idx = filenames.index(preselect) if preselect in filenames else 0
selected = st.selectbox("Select image", filenames, index=default_idx)
rec = df[df["source_filename"] == selected].iloc[0]

st.divider()

# ---- Key metrics row ----
k1, k2, k3, k4, k5, k6 = st.columns(6)
sev_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴", "critical": "🔴"}
with k1:
    st.metric("Erosion %", f"{rec['erosion_pct']:.1f}%", help=EROSION_PCT_DEFINITION)
with k2:
    st.metric("Defects", int(rec["n_defects"]))
with k3:
    sev = rec["overall_severity"]
    st.metric("Severity", f"{sev_emoji.get(sev, '')} {sev.capitalize()}",
              help=SEVERITY_BASIS)
with k4:
    st.metric("Dominant Type",
              rec["dominant_failure_type"].replace("_", " ").title())
with k5:
    conf = rec.get("mean_confidence")
    st.metric("Mean Confidence",
              f"{conf:.0%}" if conf is not None else "N/A",
              help="Mean model confidence across all detections (0–100%). "
                   "Values below 70% indicate uncertain detections that warrant human review.")
with k6:
    n_rev = int(rec["n_requires_review"])
    st.metric("Need Review", n_rev, delta="⚠" if n_rev else "✓",
              delta_color="inverse" if n_rev else "off")

st.divider()

# ---- Image quality panel (Engineering only) ----
if role == "Engineering":
    lv = rec.get("laplacian_variance")
    mb = rec.get("mean_brightness")
    qf = rec.get("quality_flag") or "unknown"
    screen_area = rec.get("screen_region_area_px", 0)
    img_w = rec.get("image_width") or rec.get("image_width", 0)
    # screen coverage from quantification_reports dimensions
    engine_tmp = create_engine(f"sqlite:///{db_path}", echo=False)
    with engine_tmp.connect() as conn:
        dim_row = conn.execute(
            text("SELECT image_width, image_height FROM quantification_reports WHERE image_id = :iid"),
            {"iid": rec["image_id"]},
        ).fetchone()
    total_px = (dim_row[0] * dim_row[1]) if dim_row and dim_row[0] and dim_row[1] else None
    screen_pct = (screen_area / total_px * 100) if (total_px and screen_area) else None

    def _focus_label(v):
        if v is None:
            return "Unknown"
        if v < 100:
            return "Poor"
        if v < 500:
            return "Acceptable"
        if v < 1500:
            return "Good"
        return "Excellent"

    def _illum_label(b):
        if b is None:
            return "Unknown"
        if b < 50:
            return "Dark"
        if b < 80:
            return "Slightly dark"
        if b < 175:
            return "Good"
        return "Overexposed"

    with st.expander("Image quality metrics", expanded=False):
        iq1, iq2, iq3, iq4 = st.columns(4)
        with iq1:
            st.metric("Focus",
                      _focus_label(lv),
                      delta=f"{lv:.0f} Laplacian variance" if lv else "not measured",
                      delta_color="off",
                      help="Laplacian variance of the image. Higher = sharper. "
                           "< 100: poor (blurry); 100–500: acceptable; > 500: good.")
        with iq2:
            st.metric("Illumination",
                      _illum_label(mb),
                      delta=f"mean brightness {mb:.0f}/255" if mb else "not measured",
                      delta_color="off",
                      help="Mean pixel brightness (0–255). 80–175 is optimal for detection accuracy.")
        with iq3:
            flag_display = qf.replace("_", " ").title() if qf != "ok" else "OK"
            st.metric("Quality flag", flag_display,
                      help="Assigned during ingestion: ok / blurry / glare / "
                           "low_resolution / occluded / bad_angle")
        with iq4:
            st.metric("Screen coverage",
                      f"{screen_pct:.0f}%" if screen_pct else "N/A",
                      help="Percentage of total image area identified as screen content. "
                           "Low values may indicate cropping or obstruction.")
        if qf not in ("ok", "unknown", None):
            st.warning(
                f"Quality flag: **{flag_display}** — detection accuracy may be reduced. "
                "Treat results with caution and verify flagged detections."
            )

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
        st.image(str(rec["annotated_path"]), use_column_width=True)
        st.caption(f"Annotated overlay — {selected}")

st.divider()

# ---- Engineering interpretation (Engineering only) ----
if role == "Engineering":
    interp = engineering_interpretation(
        dominant_failure_type=rec["dominant_failure_type"],
        erosion_pct=float(rec["erosion_pct"]),
        severity=rec["overall_severity"],
        n_requires_review=int(rec["n_requires_review"]),
        mean_confidence=rec.get("mean_confidence"),
    )

    risk_colours = {"LOW": "green", "MODERATE": "orange", "HIGH": "red", "CRITICAL": "darkred"}
    risk = interp["risk"]
    colour = risk_colours.get(risk, "grey")

    st.subheader("Engineering Assessment")
    st.markdown(
        f"**Risk:** <span style='color:{colour};font-weight:700'>{risk}</span> &nbsp;|&nbsp; "
        f"**Likely mechanism:** {interp['mechanism']}",
        unsafe_allow_html=True,
    )
    st.markdown(interp["detail"])

    # Classification evidence
    ev_text = classification_evidence(rec["dominant_failure_type"])
    st.markdown(
        f"**Classification basis** *(dominant failure type: "
        f"{rec['dominant_failure_type'].replace('_', ' ').title()})*: {ev_text}"
    )

    if interp["confidence_note"]:
        st.caption(interp["confidence_note"])

    with st.expander("Recommended actions"):
        for action in interp["actions"]:
            st.markdown(f"- {action}")

    with st.expander("Severity threshold basis"):
        st.caption(interp["severity_basis"])

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
                "Classification basis": classification_evidence(ft)[:80] + "…",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

# ---- Per-defect detail (Engineering only) ----
if role == "Engineering":
    st.divider()
    st.subheader("Per-Defect Detail")

    # ---- Scale calibration ----
    ppm = rec.get("pixels_per_mm")
    calibrated = bool(rec.get("scale_calibrated")) and ppm and float(ppm) > 0

    with st.expander(
        f"Scale calibration — {'calibrated: ' + str(round(float(ppm), 2)) + ' px/mm' if calibrated else 'not calibrated (pixel units only)'}",
        expanded=not calibrated,
    ):
        st.caption(
            "Enter the number of pixels per millimetre from a visible ruler or scale bar in the image. "
            "Once saved, all diameter and area measurements convert to physical units."
        )
        col_inp, col_btn = st.columns([2, 1])
        with col_inp:
            new_ppm = st.number_input(
                "Pixels per mm", min_value=0.1, max_value=5000.0,
                value=float(ppm) if calibrated else 10.0, step=0.1,
                key=f"ppm_{rec['image_id']}",
            )
        with col_btn:
            st.write("")
            st.write("")
            if st.button("Save calibration", key=f"save_ppm_{rec['image_id']}"):
                engine_cal = create_engine(f"sqlite:///{db_path}", echo=False)
                with engine_cal.begin() as conn:
                    conn.execute(
                        text(
                            "UPDATE quantification_reports "
                            "SET pixels_per_mm = :ppm, scale_calibrated = 1 "
                            "WHERE image_id = :iid"
                        ),
                        {"ppm": new_ppm, "iid": rec["image_id"]},
                    )
                load_summary_df.clear()
                st.rerun()

    engine_det = create_engine(f"sqlite:///{db_path}", echo=False)
    with engine_det.connect() as conn:
        det_row = conn.execute(
            text("SELECT defects_json FROM quantification_reports WHERE image_id = :iid"),
            {"iid": rec["image_id"]},
        ).fetchone()

    # Also load reasoning from classification_runs
    with engine_det.connect() as conn:
        cls_row = conn.execute(
            text("SELECT classifications_json FROM classification_runs WHERE image_id = :iid LIMIT 1"),
            {"iid": rec["image_id"]},
        ).fetchone()
    reasoning_map: dict[int, str] = {}
    if cls_row:
        for c in json.loads(cls_row[0] or "[]"):
            reasoning_map[c.get("detection_index", -1)] = c.get("reasoning", "")

    if det_row:
        defects = json.loads(det_row[0] or "[]")

        if defects:
            ppm_val = float(ppm) if calibrated else None
            areas = [d.get("defect_area_pct_of_screen", 0) for d in defects]
            diameters_px = [d.get("equivalent_diameter_px", 0) for d in defects]
            areas_px = [d.get("defect_area_px", 0) for d in defects]

            ds1, ds2, ds3, ds4 = st.columns(4)
            with ds1:
                st.metric("Largest defect", f"{max(areas):.2f}% screen area",
                          help="Area of the single largest detected defect as % of visible screen area.")
            with ds2:
                st.metric("Avg defect size", f"{sum(areas)/len(areas):.2f}% screen area",
                          help="Mean defect area across all detections.")
            with ds3:
                if ppm_val:
                    st.metric("Largest diameter",
                              f"{max(diameters_px) / ppm_val:.1f} mm",
                              help="Equivalent circular diameter of the largest defect, converted from pixels.")
                else:
                    st.metric("Largest diameter", f"{max(diameters_px):.0f} px",
                              help="Equivalent circular diameter. Calibrate scale above for mm conversion.")
            with ds4:
                if ppm_val:
                    screen_area_px = int(rec.get("screen_region_area_px", 0))
                    screen_cm2 = screen_area_px / (ppm_val ** 2) / 100
                    damaged_cm2 = sum(areas_px) / (ppm_val ** 2) / 100
                    density_cm2 = len(defects) / screen_cm2 if screen_cm2 > 0 else 0
                    st.metric("Damage density", f"{density_cm2:.1f} /cm²",
                              help="Number of detected defects per cm² of screen area.")
                else:
                    st.metric("Defect density", f"{rec.get('defect_density_per_10k_px', 0):.2f} /10k px",
                              help="Calibrate scale above for defects/cm².")

            if ppm_val:
                st.caption(
                    f"Physical scale: {ppm_val:.2f} px/mm  |  "
                    f"Total damaged area: {sum(areas_px)/(ppm_val**2)/100:.2f} cm²  |  "
                    f"Mean pit diameter: {sum(diameters_px)/len(diameters_px)/ppm_val:.1f} mm"
                )

        review_df = load_review_status_df(db_path)
        reviewed_map: dict[int, bool] = {}
        if not review_df.empty:
            for _, r in review_df[review_df["image_id"] == rec["image_id"]].iterrows():
                reviewed_map[int(r["detection_index"])] = bool(r["reviewed"])

        det_rows = []
        for d in defects:
            row_d: dict = {
                "id": f"{rec['image_id']}__{d['detection_index']}",
                "#": d["detection_index"],
                "Failure Type": d["failure_type"].replace("_", " ").title(),
                "Severity": d["severity"].capitalize(),
                "Confidence": round(d["confidence"], 2),
                "Area % screen": round(d["defect_area_pct_of_screen"], 2),
                "Diameter (px)": round(d["equivalent_diameter_px"], 1),
            }
            if calibrated and ppm_val:
                row_d["Diameter (mm)"] = round(d["equivalent_diameter_px"] / ppm_val, 2)
            row_d["Needs review"] = "⚠ Yes" if d["requires_human_review"] else "—"
            row_d["Reviewed"] = reviewed_map.get(d["detection_index"], False)
            reasoning = reasoning_map.get(d["detection_index"], "")
            if reasoning:
                row_d["Model reasoning"] = reasoning
            det_rows.append(row_d)

        det_df = pd.DataFrame(det_rows).set_index("id")
        detail_editor_key = (
            f"detail_editor_v{st.session_state.get('detail_editor_version', 0)}_{rec['image_id']}"
        )
        edited_det_df = st.data_editor(
            det_df,
            hide_index=True,
            disabled=[c for c in det_df.columns if c != "Reviewed"],
            use_container_width=True,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0, max_value=1, format="%.2f"),
                "Reviewed": st.column_config.CheckboxColumn("Reviewed"),
            },
            key=detail_editor_key,
        )

        if st.button("Save review decisions", key=f"save_review_{rec['image_id']}"):
            changed = edited_det_df.index[edited_det_df["Reviewed"] != det_df["Reviewed"]]
            if len(changed) == 0:
                st.info("No changes to save.")
            else:
                SessionFactory = ensure_classification_tables(db_path)
                with SessionFactory() as session:
                    for rid in changed:
                        image_id, det_idx = rid.rsplit("__", 1)
                        set_reviewed(session, image_id, int(det_idx),
                                     reviewed=bool(edited_det_df.loc[rid, "Reviewed"]))
                    session.commit()
                load_review_status_df.clear()
                st.session_state["detail_editor_version"] = (
                    st.session_state.get("detail_editor_version", 0) + 1
                )
                st.success(f"Saved {len(changed)} review decision(s).")
                st.rerun()

    st.divider()
    if n_rev > 0:
        st.warning(
            f"**{n_rev} detection(s) flagged for human review.** "
            "Classifier confidence below threshold — verify before use in engineering decisions."
        )
