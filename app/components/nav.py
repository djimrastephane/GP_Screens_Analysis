"""Shared sidebar header, used by every page to keep view-mode selection in sync."""

from __future__ import annotations

import streamlit as st


def render_sidebar_header() -> str:
    """Render the shared sidebar title + view-mode toggle.

    Returns the current role ('Management' or 'Engineering'). Must be called
    after st.set_page_config(). Uses a shared session_state key so the
    selection persists across page navigation instead of resetting to the
    default on every page load.
    """
    with st.sidebar:
        st.title("GP Screen Analysis")
        st.caption("Gravel Pack Failure Investigation")
        st.divider()
        role = st.radio(
            "View mode",
            ["Management", "Engineering"],
            index=1,
            key="view_mode",
        )
    return role