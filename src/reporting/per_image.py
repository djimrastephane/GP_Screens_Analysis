"""Per-image PDF report builder."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .styles import (
    CONTENT_W, CONTENT_W_L, MARGIN, PAGE_H, PAGE_W, PAGE_W_L,
    SEVERITY_COL, STYLE_BODY, STYLE_CAPTION, STYLE_CENTRE,
    STYLE_H1, STYLE_H2, STYLE_H3, STYLE_SMALL,
    TABLE_HEADER_STYLE, COL_ACCENT, COL_DARK, COL_LIGHT, COL_WHITE,
)


def _severity_cell(severity: str) -> Paragraph:
    col = SEVERITY_COL.get(severity, colors.grey)
    hex_col = "#{:02x}{:02x}{:02x}".format(
        int(col.red * 255), int(col.green * 255), int(col.blue * 255)
    )
    return Paragraph(
        f'<font color="{hex_col}"><b>{severity.upper()}</b></font>',
        STYLE_CENTRE,
    )


def _review_cell(flag: bool) -> Paragraph:
    if flag:
        return Paragraph('<font color="#E08C14"><b>⚠ YES</b></font>', STYLE_CENTRE)
    return Paragraph('<font color="#4CAF50">—</font>', STYLE_CENTRE)


def _scale_image(img_path: Path, max_w: float, max_h: float) -> RLImage | None:
    """Return a reportlab Image scaled to fit within max_w × max_h."""
    if not img_path.exists():
        return None
    from PIL import Image as PILImage
    with PILImage.open(img_path) as pil:
        iw, ih = pil.size
    ratio = min(max_w / iw, max_h / ih)
    return RLImage(str(img_path), width=iw * ratio, height=ih * ratio)


def build_per_image_report(
    quant_record,
    annotation_record,
    output_path: Path,
    generated_at: str | None = None,
) -> Path:
    """Generate a PDF report for one GP screen image.

    Args:
        quant_record: QuantificationRecord ORM row.
        annotation_record: AnnotationRecord ORM row (or None).
        output_path: destination PDF path.
        generated_at: ISO timestamp string; defaults to now.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if generated_at is None:
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=(PAGE_W, PAGE_H),
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title=f"GP Screen Failure Report – {quant_record.source_filename}",
        author="GP Screens Analysis",
    )

    story = []

    # ------------------------------------------------------------------
    # Page 1 — Cover / Summary
    # ------------------------------------------------------------------
    story.append(Paragraph("GP Screen Failure Analysis Report", STYLE_H1))
    story.append(Paragraph(f"Image: <b>{quant_record.source_filename}</b>", STYLE_BODY))
    story.append(Paragraph(f"Generated: {generated_at}", STYLE_SMALL))
    story.append(Spacer(1, 0.3 * cm))

    # Key metrics table
    sev_col = SEVERITY_COL.get(quant_record.overall_severity, colors.grey)
    metrics_data = [
        ["Metric", "Value"],
        ["Erosion % (screen region)", f"{quant_record.erosion_pct:.2f} %"],
        ["Overall Severity", quant_record.overall_severity.upper()],
        ["Dominant Failure Type", quant_record.dominant_failure_type.replace("_", " ").title()],
        ["Total Defects", str(quant_record.n_defects)],
        ["Require Human Review", str(quant_record.n_requires_review)],
        ["Scale Calibrated", "Yes" if quant_record.scale_calibrated else "No"],
        ["Screen Region Area (px)", f"{quant_record.screen_region_area_px:,}"],
        ["Composite Defect Area (px)", f"{quant_record.composite_defect_area_px:,}"],
        ["Defect Density (per 10k px)", f"{quant_record.defect_density_per_10k_px:.4f}"],
    ]
    metrics_table = Table(metrics_data, colWidths=[CONTENT_W * 0.52, CONTENT_W * 0.48])
    metrics_table.setStyle(TABLE_HEADER_STYLE)
    # Highlight severity row
    for i, row in enumerate(metrics_data):
        if row[0] == "Overall Severity":
            metrics_table.setStyle(TableStyle([
                ("TEXTCOLOR", (1, i), (1, i), sev_col),
                ("FONTNAME",  (1, i), (1, i), "Helvetica-Bold"),
            ]))
    story.append(metrics_table)
    story.append(Spacer(1, 0.4 * cm))

    # Failure type breakdown
    ft_breakdown = json.loads(quant_record.failure_type_breakdown_json or "{}")
    if ft_breakdown:
        story.append(Paragraph("Failure Type Breakdown", STYLE_H2))
        ft_data = [["Failure Type", "Count", "Area (px)", "Area (% screen)"]]
        for ft, stats in sorted(ft_breakdown.items(), key=lambda x: -x[1]["count"]):
            ft_data.append([
                ft.replace("_", " ").title(),
                str(stats["count"]),
                f"{stats['area_px']:,}",
                f"{stats['area_pct_of_screen']:.2f} %",
            ])
        ft_table = Table(
            ft_data,
            colWidths=[CONTENT_W * 0.40, CONTENT_W * 0.15,
                       CONTENT_W * 0.25, CONTENT_W * 0.20],
        )
        ft_table.setStyle(TABLE_HEADER_STYLE)
        story.append(ft_table)
        story.append(Spacer(1, 0.3 * cm))

    # ------------------------------------------------------------------
    # Page 2 — Annotated overlay image
    # ------------------------------------------------------------------
    if annotation_record and annotation_record.annotated_path:
        story.append(PageBreak())
        story.append(Paragraph("Annotated Overlay", STYLE_H2))
        img = _scale_image(
            Path(annotation_record.annotated_path),
            CONTENT_W, PAGE_H - 4 * MARGIN,
        )
        if img:
            story.append(img)
            story.append(Paragraph(
                f"Figure 1. Annotated overlay — {quant_record.source_filename}. "
                "Coloured regions indicate detected failure types. "
                "Dashed boxes indicate detections flagged for human review.",
                STYLE_CAPTION,
            ))

    # ------------------------------------------------------------------
    # Page 3 — 3-panel comparison
    # ------------------------------------------------------------------
    if annotation_record and annotation_record.panel_path:
        story.append(PageBreak())
        story.append(Paragraph("Three-Panel Comparison", STYLE_H2))
        story.append(Paragraph(
            "Left: Original preprocessed image.  "
            "Centre: Annotated overlay with failure type classification.  "
            "Right: Binary composite defect mask.",
            STYLE_BODY,
        ))
        story.append(Spacer(1, 0.2 * cm))
        panel_img = _scale_image(
            Path(annotation_record.panel_path),
            CONTENT_W, (PAGE_H - 4 * MARGIN) * 0.55,
        )
        if panel_img:
            story.append(panel_img)

    # ------------------------------------------------------------------
    # Page 4 — Per-defect detail table
    # ------------------------------------------------------------------
    defects = json.loads(quant_record.defects_json or "[]")
    if defects:
        story.append(PageBreak())
        story.append(Paragraph("Per-Defect Detail", STYLE_H2))

        det_data = [["#", "Failure Type", "Severity", "Confidence",
                      "Area %", "Diameter (px)", "Review"]]
        for d in defects:
            det_data.append([
                str(d["detection_index"]),
                d["failure_type"].replace("_", " ").title(),
                _severity_cell(d["severity"]),
                f"{d['confidence']:.2f}",
                f"{d['defect_area_pct_of_screen']:.2f}",
                f"{d['equivalent_diameter_px']:.1f}",
                _review_cell(d["requires_human_review"]),
            ])

        det_table = Table(
            det_data,
            colWidths=[
                CONTENT_W * 0.06, CONTENT_W * 0.25, CONTENT_W * 0.12,
                CONTENT_W * 0.12, CONTENT_W * 0.11, CONTENT_W * 0.16,
                CONTENT_W * 0.18,
            ],
        )
        det_table.setStyle(TABLE_HEADER_STYLE)
        story.append(det_table)
        story.append(Spacer(1, 0.4 * cm))

        # Notes
        story.append(Paragraph("Notes", STYLE_H3))
        if not quant_record.scale_calibrated:
            story.append(Paragraph(
                "⚠ Scale not calibrated. Diameter measurements are in pixels only. "
                "Set <i>pixels_per_mm</i> in severity_config.yaml to enable physical measurements.",
                STYLE_SMALL,
            ))
        if quant_record.n_requires_review:
            story.append(Paragraph(
                f"⚠ {quant_record.n_requires_review} detection(s) flagged for human review "
                "(classifier confidence below threshold). Results for these detections should "
                "be verified by an engineer before use in decision-making.",
                STYLE_SMALL,
            ))

    doc.build(story)
    return output_path
