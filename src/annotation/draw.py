"""Drawing primitives: mask overlays, boxes, labels, legend, header bar."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .colours import (
    HEADER_BG, HEADER_FG, HEADER_H,
    LEGEND_BG, LEGEND_FG, LEGEND_SWATCH_SIZE,
    MASK_ALPHA, REVIEW_BADGE_BG, REVIEW_BADGE_FG,
    SEVERITY_THICKNESS, get_colour, get_severity_colour, rgb_to_bgr,
)

# Font sizes
_LABEL_FONT_SIZE  = 12
_HEADER_FONT_SIZE = 13
_LEGEND_FONT_SIZE = 11

# Try to load a monospaced/clean system font; fallback to PIL default
def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


_LABEL_FONT  = _load_font(_LABEL_FONT_SIZE)
_HEADER_FONT = _load_font(_HEADER_FONT_SIZE)
_LEGEND_FONT = _load_font(_LEGEND_FONT_SIZE)


# ---------------------------------------------------------------------------
# Mask overlay
# ---------------------------------------------------------------------------

def overlay_mask(
    canvas: np.ndarray,
    mask: np.ndarray,
    colour_rgb: tuple[int, int, int],
    alpha: float = MASK_ALPHA,
) -> np.ndarray:
    """Blend a coloured fill over mask pixels in-place on canvas (uint8 RGB)."""
    if not mask.any():
        return canvas
    fill = np.zeros_like(canvas, dtype=np.uint8)
    fill[mask > 0] = colour_rgb
    canvas = canvas.astype(np.float32)
    fg = mask > 0
    canvas[fg] = canvas[fg] * (1 - alpha) + fill[fg] * alpha
    return np.clip(canvas, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Bounding box
# ---------------------------------------------------------------------------

def draw_box(
    img_bgr: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    colour_rgb: tuple[int, int, int],
    thickness: int = 2,
    dashed: bool = False,
) -> None:
    """Draw a rectangle on a BGR OpenCV image (in-place)."""
    colour_bgr = rgb_to_bgr(colour_rgb)
    if not dashed:
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), colour_bgr, thickness)
        return
    # Dashed box — draw short segments
    dash = 8
    for x in range(x1, x2, dash * 2):
        cv2.line(img_bgr, (x, y1), (min(x + dash, x2), y1), colour_bgr, thickness)
        cv2.line(img_bgr, (x, y2), (min(x + dash, x2), y2), colour_bgr, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(img_bgr, (x1, y), (x1, min(y + dash, y2)), colour_bgr, thickness)
        cv2.line(img_bgr, (x2, y), (x2, min(y + dash, y2)), colour_bgr, thickness)


# ---------------------------------------------------------------------------
# Defect label
# ---------------------------------------------------------------------------

def draw_defect_label(
    pil_img: Image.Image,
    x1: int, y1: int, x2: int,
    failure_type: str,
    severity: str,
    confidence: float,
    area_pct: float,
    requires_review: bool,
) -> None:
    """Draw a two-line label box above a detection bbox on a PIL Image (in-place)."""
    draw = ImageDraw.Draw(pil_img)
    col = get_colour(failure_type)
    sev_col = get_severity_colour(severity)

    line1 = f"{failure_type.replace('_', ' ')}"
    line2 = f"conf:{confidence:.2f}  area:{area_pct:.1f}%"

    # Measure text
    bbox1 = draw.textbbox((0, 0), line1, font=_LABEL_FONT)
    bbox2 = draw.textbbox((0, 0), line2, font=_LABEL_FONT)
    tw = max(bbox1[2] - bbox1[0], bbox2[2] - bbox2[0]) + 6
    th = (bbox1[3] - bbox1[1]) + (bbox2[3] - bbox2[1]) + 8

    # Position: above bbox, clamp to image top
    lx = max(0, min(x1, pil_img.width - tw - 2))
    ly = max(0, y1 - th - 2)

    # Background swatch
    draw.rectangle([lx, ly, lx + tw, ly + th], fill=col + (210,) if len(col) == 3 else col)

    # Severity dot
    dot_r = 4
    dot_x = lx + tw - dot_r - 3
    dot_y = ly + (bbox1[3] - bbox1[1]) // 2 + 3
    draw.ellipse([dot_x - dot_r, dot_y - dot_r, dot_x + dot_r, dot_y + dot_r], fill=sev_col)

    # Text
    draw.text((lx + 3, ly + 2), line1, font=_LABEL_FONT, fill=(255, 255, 255))
    draw.text((lx + 3, ly + (bbox1[3] - bbox1[1]) + 4), line2, font=_LABEL_FONT, fill=(220, 220, 220))

    # Review badge
    if requires_review:
        bw, bh = 14, 14
        bx, by = x2 - bw - 1, y1 + 1
        bx = max(0, min(bx, pil_img.width - bw - 1))
        by = max(0, by)
        draw.rectangle([bx, by, bx + bw, by + bh], fill=REVIEW_BADGE_BG)
        draw.text((bx + 3, by + 1), "?", font=_LABEL_FONT, fill=REVIEW_BADGE_FG)


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------

def draw_legend(
    pil_img: Image.Image,
    failure_types: list[str],   # unique types present, with counts
    type_counts: dict[str, int],
) -> None:
    """Draw a legend panel in the bottom-right corner of a PIL Image (in-place)."""
    if not failure_types:
        return

    draw = ImageDraw.Draw(pil_img)
    line_h = _LEGEND_FONT_SIZE + 6
    pad = 6
    title = "Legend"
    title_bbox = draw.textbbox((0, 0), title, font=_LEGEND_FONT)
    title_w = title_bbox[2] - title_bbox[0]

    max_text_w = max(
        (draw.textbbox((0, 0), f"  {ft.replace('_',' ')} ({type_counts.get(ft,0)})", font=_LEGEND_FONT)[2] for ft in failure_types),
        default=0,
    )
    legend_w = max(title_w + 2 * pad, LEGEND_SWATCH_SIZE + max_text_w + 3 * pad)
    legend_h = (len(failure_types) + 1) * line_h + 2 * pad

    iw, ih = pil_img.size
    lx = iw - legend_w - 8
    ly = ih - legend_h - 8
    lx, ly = max(0, lx), max(0, ly)

    # Background
    draw.rectangle([lx, ly, lx + legend_w, ly + legend_h], fill=LEGEND_BG + (200,) if False else LEGEND_BG)

    # Title
    draw.text((lx + pad, ly + pad), title, font=_LEGEND_FONT, fill=LEGEND_FG)

    for i, ft in enumerate(failure_types):
        row_y = ly + pad + (i + 1) * line_h
        colour = get_colour(ft)
        sx, sy = lx + pad, row_y + 2
        draw.rectangle([sx, sy, sx + LEGEND_SWATCH_SIZE, sy + LEGEND_SWATCH_SIZE], fill=colour)
        label = f"  {ft.replace('_', ' ')} ({type_counts.get(ft, 0)})"
        draw.text((sx + LEGEND_SWATCH_SIZE + 3, row_y), label, font=_LEGEND_FONT, fill=LEGEND_FG)


# ---------------------------------------------------------------------------
# Header bar
# ---------------------------------------------------------------------------

def draw_header_bar(
    source_filename: str,
    erosion_pct: float,
    n_defects: int,
    overall_severity: str,
    n_requires_review: int,
    width: int,
) -> np.ndarray:
    """Return a uint8 RGB header bar of shape (HEADER_H, width, 3)."""
    bar = Image.new("RGB", (width, HEADER_H), HEADER_BG)
    draw = ImageDraw.Draw(bar)

    sev_col = get_severity_colour(overall_severity)
    review_str = f"  ⚠ {n_requires_review} review" if n_requires_review else ""
    text = (
        f"  {source_filename}   |   Erosion: {erosion_pct:.1f}%"
        f"   |   {n_defects} defect{'s' if n_defects != 1 else ''}"
        f"   |   Severity: {overall_severity.upper()}{review_str}"
    )
    draw.text((4, (HEADER_H - _HEADER_FONT_SIZE) // 2), text, font=_HEADER_FONT, fill=HEADER_FG)

    # Severity colour accent on left edge
    draw.rectangle([0, 0, 4, HEADER_H], fill=sev_col)

    return np.array(bar, dtype=np.uint8)
