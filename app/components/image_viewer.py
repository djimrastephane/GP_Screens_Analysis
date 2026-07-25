"""Image display helpers for the Streamlit dashboard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image


def _load(path: str | Path | None) -> Image.Image | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return Image.open(p).convert("RGB")


def show_three_panel(
    preprocessed_path: str | None,
    annotated_path: str | None,
    mask_path: str | None,
    captions: tuple[str, str, str] = ("Original", "Annotated", "Mask"),
) -> None:
    """Display original | annotated overlay | binary mask in three columns."""
    col1, col2, col3 = st.columns(3)
    for col, path, caption in zip(
        [col1, col2, col3],
        [preprocessed_path, annotated_path, mask_path],
        captions,
    ):
        with col:
            img = _load(path)
            if img:
                st.image(img, caption=caption, width="stretch")
            else:
                st.info(f"{caption}: not available")


def show_annotated(path: str | None, caption: str = "") -> None:
    """Display the annotated overlay full-width."""
    img = _load(path)
    if img:
        st.image(img, caption=caption, width="stretch")
    else:
        st.info("Annotated image not available")


def show_panel(path: str | None) -> None:
    """Display the 3-panel composite image (Original | Annotated | Mask)."""
    img = _load(path)
    if img:
        st.image(img, width="stretch")
    else:
        st.info("Panel image not available")


def severity_badge(severity: str) -> str:
    """Return a coloured HTML badge string for inline display."""
    colours = {
        "low": ("#2ea82e", "#e8f5e8"),
        "medium": ("#c47c00", "#fff3cd"),
        "high": ("#c03000", "#ffe0cc"),
        "critical": ("#900000", "#ffd0d0"),
    }
    fg, bg = colours.get(severity, ("#555", "#eee"))
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 8px;'
        f'border-radius:4px;font-weight:600;font-size:0.82em;">'
        f'{severity.upper()}</span>'
    )
