"""Compose single annotated images and multi-panel layouts."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .colours import SEVERITY_THICKNESS, get_colour, rgb_to_bgr
from .draw import (
    draw_box,
    draw_defect_label,
    draw_header_bar,
    draw_legend,
    overlay_mask,
)


def compose_annotated(
    img: np.ndarray,
    detections: list[dict],
    defects: list[dict],
    masks: list[np.ndarray],
    source_filename: str,
    erosion_pct: float,
    overall_severity: str,
    n_requires_review: int,
    include_header: bool = True,
    include_legend: bool = True,
) -> np.ndarray:
    """Produce a fully annotated image from pipeline data.

    Args:
        img: uint8 RGB preprocessed image.
        detections: list of detection dicts (x1,y1,x2,y2,class_name,…).
        defects: list of DefectMeasurement.to_dict() entries (parallel with detections).
        masks: list of HxW uint8 binary masks (parallel with detections).
        source_filename: for header bar.
        erosion_pct: for header bar.
        overall_severity: for header bar and styling.
        n_requires_review: for header bar.
        include_header: whether to add the header bar strip.
        include_legend: whether to draw the legend panel.

    Returns:
        uint8 RGB annotated image.
    """
    canvas = img.copy()

    # --- Mask fills ---
    for defect, mask in zip(defects, masks):
        colour = get_colour(defect["failure_type"])
        canvas = overlay_mask(canvas, mask, colour)

    # --- Bounding boxes (OpenCV, BGR) ---
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    for det, defect in zip(detections, defects):
        colour = get_colour(defect["failure_type"])
        thickness = SEVERITY_THICKNESS.get(defect["severity"], 2)
        dashed = defect.get("requires_human_review", False)
        draw_box(
            canvas_bgr,
            int(det["x1"]), int(det["y1"]),
            int(det["x2"]), int(det["y2"]),
            colour, thickness, dashed,
        )
    canvas = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)

    # --- Labels and legend (PIL) ---
    pil = Image.fromarray(canvas)
    for det, defect in zip(detections, defects):
        draw_defect_label(
            pil,
            int(det["x1"]), int(det["y1"]), int(det["x2"]),
            defect["failure_type"],
            defect["severity"],
            defect["confidence"],
            defect.get("defect_area_pct_of_screen", 0.0),
            defect.get("requires_human_review", False),
        )

    if include_legend:
        type_counts: dict[str, int] = {}
        for d in defects:
            ft = d["failure_type"]
            type_counts[ft] = type_counts.get(ft, 0) + 1
        draw_legend(pil, list(type_counts.keys()), type_counts)

    canvas = np.array(pil, dtype=np.uint8)

    # --- Header bar ---
    if include_header:
        header = draw_header_bar(
            source_filename, erosion_pct,
            len(defects), overall_severity,
            n_requires_review, canvas.shape[1],
        )
        canvas = np.vstack([header, canvas])

    return canvas


def compose_panels(
    original: np.ndarray,
    annotated: np.ndarray,
    composite_mask: np.ndarray,
    header: np.ndarray | None = None,
) -> np.ndarray:
    """Create a side-by-side three-panel layout: Original | Annotated | Mask.

    Args:
        original: uint8 RGB source image.
        annotated: uint8 RGB annotated image (without header strip).
        composite_mask: HxW uint8 binary mask.
        header: optional header bar to prepend (width must match 3×panel_w).

    Returns:
        uint8 RGB panel image.
    """
    h, w = original.shape[:2]

    # Mask to RGB for display
    mask_rgb = np.stack([composite_mask] * 3, axis=-1)

    panel = np.hstack([original, annotated[:h, :w], mask_rgb])

    if header is not None:
        full_w = panel.shape[1]
        if header.shape[1] != full_w:
            # Re-draw header at correct width
            from .draw import draw_header_bar as _hb
            # Just resize — header content already drawn
            header_resized = cv2.resize(
                cv2.cvtColor(header, cv2.COLOR_RGB2BGR),
                (full_w, header.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            header = cv2.cvtColor(header_resized, cv2.COLOR_BGR2RGB)
        panel = np.vstack([header, panel])

    return panel
