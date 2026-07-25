"""Image gallery — browse annotated overlays, tag wells, and compare screens."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

from app.components.nav import render_sidebar_header
from app.components.data_loader import get_db_path, load_summary_df
from app.components.charts import erosion_bar

st.set_page_config(page_title="Gallery", page_icon="🖼️", layout="wide")

role = render_sidebar_header()

with st.sidebar:
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
    st.divider()
    group_by_well = st.toggle("Group by well", value=False, key="gallery_group_by_well")

db_path = get_db_path()
df_all = load_summary_df(db_path)

st.title("Image Gallery")

# ---- Well / zone tagging (operates on all images, independent of filters below) ----
with st.expander("Tag wells (optional)"):
    st.caption(
        "Assign a well name and completion zone to each image so screens from the same "
        "well can be grouped and compared. Leave blank to clear a tag."
    )
    tag_source = df_all[["image_id", "source_filename", "well_name", "completion_zone"]].copy()
    tag_source = tag_source.rename(columns={
        "source_filename": "Filename", "well_name": "Well", "completion_zone": "Zone",
    }).set_index("image_id")
    tag_source[["Well", "Zone"]] = tag_source[["Well", "Zone"]].fillna("")

    tag_editor_key = f"well_tag_editor_v{st.session_state.get('well_tag_editor_version', 0)}"
    edited_tags = st.data_editor(
        tag_source,
        hide_index=True,
        disabled=["Filename"],
        use_container_width=True,
        key=tag_editor_key,
    )

    if st.button("Save well tags"):
        changed_mask = (
            (edited_tags["Well"].fillna("") != tag_source["Well"])
            | (edited_tags["Zone"].fillna("") != tag_source["Zone"])
        )
        changed_ids = edited_tags.index[changed_mask]
        if len(changed_ids) == 0:
            st.info("No changes to save.")
        else:
            engine = create_engine(f"sqlite:///{db_path}", echo=False)
            with engine.begin() as conn:
                for image_id in changed_ids:
                    well = edited_tags.loc[image_id, "Well"] or None
                    zone = edited_tags.loc[image_id, "Zone"] or None
                    conn.execute(
                        text(
                            "UPDATE image_metadata SET well_name = :well, completion_zone = :zone "
                            "WHERE image_id = :iid"
                        ),
                        {"well": well, "zone": zone, "iid": image_id},
                    )
            load_summary_df.clear()
            st.session_state["well_tag_editor_version"] = (
                st.session_state.get("well_tag_editor_version", 0) + 1
            )
            st.success(f"Tagged {len(changed_ids)} image(s).")
            st.rerun()

st.divider()

df = df_all.copy()
if severity_filter:
    df = df[df["overall_severity"].isin(severity_filter)]
if failure_filter:
    df = df[df["dominant_failure_type"].isin(failure_filter)]

st.caption(f"Showing {len(df)} image{'s' if len(df) != 1 else ''}")

if df.empty:
    st.info("No images match the current filters.")
    st.stop()

# ---- Sort options ----
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
sev_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴", "critical": "🔴"}


def _render_card(col, rec) -> None:
    with col:
        ann_path = rec.get("annotated_path")
        if ann_path and Path(ann_path).exists():
            st.image(str(ann_path), use_container_width=True)
        else:
            st.info("Image not available")

        st.markdown(f"**{rec['source_filename']}**")
        st.checkbox("Compare", key=f"cmp_{rec['image_id']}")

        sev = rec["overall_severity"]
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;"
            f"font-size:0.95rem;padding:4px 0'>"
            f"<span><b>{rec['erosion_pct']:.1f}%</b><br>"
            f"<span style='color:#888;font-size:0.75rem'>Erosion</span></span>"
            f"<span><b>{int(rec['n_defects'])}</b><br>"
            f"<span style='color:#888;font-size:0.75rem'>Defects</span></span>"
            f"<span><b>{sev_emoji.get(sev,'')} {sev.capitalize()}</b><br>"
            f"<span style='color:#888;font-size:0.75rem'>Severity</span></span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        if role == "Engineering":
            dom = rec["dominant_failure_type"].replace("_", " ").title()
            st.caption(f"Dominant: {dom}")
            if rec["n_requires_review"] > 0:
                st.warning(f"⚠ {int(rec['n_requires_review'])} detection(s) need review")


if group_by_well:
    grouped = df.copy()
    grouped["_well"] = grouped["well_name"].fillna("Unassigned")
    for well, well_df in grouped.groupby("_well", sort=True):
        head_col, btn_col = st.columns([5, 1])
        with head_col:
            st.subheader(well)
        with btn_col:
            safe_key = "".join(c if c.isalnum() else "_" for c in well)
            if st.button("Compare this well →", key=f"cmpwell_{safe_key}"):
                for iid in well_df["image_id"]:
                    st.session_state[f"cmp_{iid}"] = True
                st.rerun()

        for row_start in range(0, len(well_df), cols_per_row):
            row_df = well_df.iloc[row_start: row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, (_, rec) in zip(cols, row_df.iterrows()):
                _render_card(col, rec)
        st.divider()
else:
    for row_start in range(0, len(df), cols_per_row):
        row_df = df.iloc[row_start: row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, (_, rec) in zip(cols, row_df.iterrows()):
            _render_card(col, rec)
        st.divider()

# ---- Comparison ----
compare_selection = {
    iid for iid in df_all["image_id"]
    if st.session_state.get(f"cmp_{iid}", False)
}

if len(compare_selection) >= 2:
    compare_df = df_all[df_all["image_id"].isin(compare_selection)]
    st.subheader(f"Compare {len(compare_selection)} selected screens")

    table_cols = ["source_filename", "erosion_pct", "n_defects", "overall_severity",
                  "dominant_failure_type", "mean_confidence", "well_name"]
    table_cols = [c for c in table_cols if c in compare_df.columns]
    table = compare_df[table_cols].rename(columns={
        "source_filename": "Filename", "erosion_pct": "Erosion %", "n_defects": "Defects",
        "overall_severity": "Severity", "dominant_failure_type": "Dominant Failure",
        "mean_confidence": "Mean Confidence", "well_name": "Well",
    })
    st.dataframe(table, hide_index=True, use_container_width=True)

    thumb_cols = st.columns(len(compare_selection))
    for col, (_, rec) in zip(thumb_cols, compare_df.iterrows()):
        with col:
            ann_path = rec.get("annotated_path")
            if ann_path and Path(ann_path).exists():
                st.image(str(ann_path), caption=rec["source_filename"], use_container_width=True)

    st.plotly_chart(erosion_bar(compare_df), use_container_width=True)
elif len(compare_selection) == 1:
    st.caption("Select at least one more screen to compare.")
