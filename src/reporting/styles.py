"""Page layout constants and reportlab style factories."""

from __future__ import annotations

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm

# ---------------------------------------------------------------------------
# Page geometry
# ---------------------------------------------------------------------------

PAGE_W, PAGE_H = A4
PAGE_W_L, PAGE_H_L = landscape(A4)
MARGIN = 1.8 * cm
CONTENT_W = PAGE_W - 2 * MARGIN
CONTENT_W_L = PAGE_W_L - 2 * MARGIN
CONTENT_H = PAGE_H - 2 * MARGIN

# ---------------------------------------------------------------------------
# Brand colours (reportlab Color objects)
# ---------------------------------------------------------------------------

COL_DARK    = colors.Color(0.12, 0.12, 0.12)
COL_MID     = colors.Color(0.35, 0.35, 0.35)
COL_LIGHT   = colors.Color(0.88, 0.88, 0.88)
COL_ACCENT  = colors.Color(0.10, 0.35, 0.55)   # dark teal/blue
COL_WHITE   = colors.white
COL_RULE    = colors.Color(0.75, 0.75, 0.75)

SEVERITY_COL = {
    "low":      colors.Color(0.20, 0.65, 0.20),
    "medium":   colors.Color(0.88, 0.60, 0.08),
    "high":     colors.Color(0.85, 0.28, 0.08),
    "critical": colors.Color(0.75, 0.05, 0.05),
}

# Failure-type matplotlib / reportlab hex colours (matching annotation module)
FAILURE_TYPE_HEX = {
    "erosion_hole":       "#DC3737",
    "corrosion_pitting":  "#E69B19",
    "wire_wrap_failure":  "#28B4DC",
    "screen_collapse":    "#AF23AF",
    "mechanical_damage":  "#5A6EDC",
    "plugging_partial":   "#2DC35F",
    "plugging_complete":  "#0F5F32",
    "unknown":            "#969696",
}


# ---------------------------------------------------------------------------
# Paragraph styles
# ---------------------------------------------------------------------------

_BASE = getSampleStyleSheet()

STYLE_H1 = ParagraphStyle(
    "H1", parent=_BASE["Heading1"],
    fontName="Helvetica-Bold", fontSize=18, textColor=COL_DARK,
    spaceAfter=6, spaceBefore=4,
)
STYLE_H2 = ParagraphStyle(
    "H2", parent=_BASE["Heading2"],
    fontName="Helvetica-Bold", fontSize=13, textColor=COL_ACCENT,
    spaceAfter=4, spaceBefore=8,
)
STYLE_H3 = ParagraphStyle(
    "H3", parent=_BASE["Heading3"],
    fontName="Helvetica-Bold", fontSize=10, textColor=COL_MID,
    spaceAfter=2, spaceBefore=4,
)
STYLE_BODY = ParagraphStyle(
    "Body", parent=_BASE["Normal"],
    fontName="Helvetica", fontSize=9, textColor=COL_DARK,
    spaceAfter=3, leading=13,
)
STYLE_SMALL = ParagraphStyle(
    "Small", parent=_BASE["Normal"],
    fontName="Helvetica", fontSize=8, textColor=COL_MID,
    spaceAfter=2,
)
STYLE_CAPTION = ParagraphStyle(
    "Caption", parent=_BASE["Normal"],
    fontName="Helvetica-Oblique", fontSize=8, textColor=COL_MID,
    alignment=TA_CENTER, spaceAfter=4,
)
STYLE_MONO = ParagraphStyle(
    "Mono", parent=_BASE["Normal"],
    fontName="Courier", fontSize=8, textColor=COL_DARK,
)
STYLE_CENTRE = ParagraphStyle(
    "Centre", parent=_BASE["Normal"],
    fontName="Helvetica", fontSize=9, textColor=COL_DARK,
    alignment=TA_CENTER,
)

# ---------------------------------------------------------------------------
# Table styles
# ---------------------------------------------------------------------------

from reportlab.platypus import TableStyle

TABLE_HEADER_STYLE = TableStyle([
    ("BACKGROUND",  (0, 0), (-1, 0), COL_ACCENT),
    ("TEXTCOLOR",   (0, 0), (-1, 0), COL_WHITE),
    ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE",    (0, 0), (-1, -1), 8),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.Color(0.95, 0.95, 0.97)]),
    ("GRID",        (0, 0), (-1, -1), 0.25, COL_RULE),
    ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
    ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ("RIGHTPADDING",(0, 0), (-1, -1), 4),
    ("TOPPADDING",  (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
])

SUMMARY_TABLE_STYLE = TableStyle([
    ("BACKGROUND",  (0, 0), (-1, 0), COL_DARK),
    ("TEXTCOLOR",   (0, 0), (-1, 0), COL_WHITE),
    ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE",    (0, 0), (-1, -1), 8),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.Color(0.94, 0.94, 0.96)]),
    ("GRID",        (0, 0), (-1, -1), 0.3, COL_RULE),
    ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
    ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ("RIGHTPADDING",(0, 0), (-1, -1), 5),
    ("TOPPADDING",  (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
])
