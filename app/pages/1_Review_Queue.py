"""Review Queue — triage worklist for detections flagged for human review."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.components.nav import render_sidebar_header
from app.components.data_loader import get_db_path, load_defects_df, load_review_status_df
from src.classification.store import ensure_table as ensure_classification_tables, set_reviewed

st.set_page_config(page_title="Review Queue", page_icon="🔎", layout="wide")

role = render_sidebar_header()

db_path = get_db_path()
ensure_classification_tables(db_path)   # creates review_status table if missing

defects_df = load_defects_df(db_path)
review_df = load_review_status_df(db_path)

st.title("Review Queue")
st.caption("Detections flagged for human review — least confident first.")
st.divider()

if defects_df.empty:
    st.info("No detections found. Run the pipeline first.")
    st.stop()

queue = defects_df[defects_df["requires_human_review"] == True].copy()
queue["id"] = queue["image_id"] + "__" + queue["detection_index"].astype(str)
queue = queue.merge(review_df[["id", "reviewed"]], on="id", how="left")
queue["reviewed"] = queue["reviewed"].fillna(False).astype(bool)

with st.sidebar:
    st.divider()
    show_reviewed = st.toggle("Show already-reviewed items", value=False, key="rq_show_reviewed")
    sev_options = sorted(queue["severity"].dropna().unique().tolist())
    sev_filter = st.multiselect("Severity", sev_options, default=[], key="rq_sev_filter")
    ft_options = sorted(queue["failure_type"].dropna().unique().tolist())
    ft_filter = st.multiselect("Failure type", ft_options, default=[], key="rq_ft_filter")

if not show_reviewed:
    queue = queue[~queue["reviewed"]]
if sev_filter:
    queue = queue[queue["severity"].isin(sev_filter)]
if ft_filter:
    queue = queue[queue["failure_type"].isin(ft_filter)]

sev_rank = {"critical": 3, "high": 2, "medium": 1, "low": 0}
queue["_sev_rank"] = queue["severity"].map(sev_rank).fillna(0)
queue = queue.sort_values(["_sev_rank", "confidence"], ascending=[False, True]).drop(columns="_sev_rank")

st.caption(f"{len(queue)} detection(s) in view")

if queue.empty:
    st.success("Nothing to review.")
    st.stop()

display_cols = {
    "source_filename": "Filename",
    "failure_type": "Failure Type",
    "severity": "Severity",
    "confidence": "Confidence",
    "reasoning": "Model Reasoning",
}
display = queue[["id", *display_cols.keys(), "reviewed"]].rename(columns=display_cols)
display = display.rename(columns={"reviewed": "Reviewed"})
display["Failure Type"] = display["Failure Type"].str.replace("_", " ").str.title()
display["Severity"] = display["Severity"].str.capitalize()
display = display.set_index("id")

editor_key = f"review_editor_v{st.session_state.get('review_editor_version', 0)}"
edited = st.data_editor(
    display,
    hide_index=True,
    disabled=[c for c in display.columns if c != "Reviewed"],
    column_config={
        "Confidence": st.column_config.ProgressColumn(
            "Confidence", min_value=0, max_value=1, format="%.2f"),
        "Reviewed": st.column_config.CheckboxColumn("Reviewed"),
    },
    key=editor_key,
    use_container_width=True,
)

if st.button("Save review decisions", type="primary"):
    changed = edited.index[edited["Reviewed"] != display["Reviewed"]]
    if len(changed) == 0:
        st.info("No changes to save.")
    else:
        SessionFactory = ensure_classification_tables(db_path)
        with SessionFactory() as session:
            for rid in changed:
                image_id, det_idx = rid.rsplit("__", 1)
                set_reviewed(session, image_id, int(det_idx), reviewed=bool(edited.loc[rid, "Reviewed"]))
            session.commit()
        load_review_status_df.clear()
        st.session_state["review_editor_version"] = st.session_state.get("review_editor_version", 0) + 1
        st.success(f"Saved {len(changed)} review decision(s).")
        st.rerun()

st.divider()
st.subheader("Jump to image")
for filename in queue["source_filename"].drop_duplicates():
    st.page_link(
        "pages/3_Analysis.py",
        label=filename,
        icon="🔍",
        query_params={"image": filename},
    )
