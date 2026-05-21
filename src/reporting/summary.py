"""Campaign summary PDF report builder."""

from __future__ import annotations

import json
import tempfile
from collections import Counter
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
)

from .charts import erosion_bar_chart, failure_type_chart, severity_pie_chart
from .styles import (
    CONTENT_W, MARGIN, PAGE_H, PAGE_W,
    SEVERITY_COL, STYLE_BODY, STYLE_CAPTION, STYLE_H1, STYLE_H2,
    STYLE_SMALL, SUMMARY_TABLE_STYLE, TABLE_HEADER_STYLE, COL_WHITE,
)


def _scale_chart(path: Path, max_w: float, max_h: float) -> RLImage | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    from PIL import Image as PILImage
    with PILImage.open(path) as p:
        iw, ih = p.size
    ratio = min(max_w / iw, max_h / ih)
    return RLImage(str(path), width=iw * ratio, height=ih * ratio)


def _scale_thumbnail(img_path: Path, max_w: float, max_h: float) -> RLImage | None:
    if not img_path or not img_path.exists():
        return None
    from PIL import Image as PILImage
    with PILImage.open(img_path) as p:
        iw, ih = p.size
    ratio = min(max_w / iw, max_h / ih)
    return RLImage(str(img_path), width=iw * ratio, height=ih * ratio)


def build_summary_report(
    quant_records: list,
    annotation_map: dict,    # {image_id: AnnotationRecord}
    charts_dir: Path,
    output_path: Path,
    generated_at: str | None = None,
) -> Path:
    """Generate a campaign summary PDF across all analysed images.

    Args:
        quant_records: list of QuantificationRecord ORM rows.
        annotation_map: map from image_id to AnnotationRecord.
        charts_dir: writable directory for intermediate chart PNGs.
        output_path: destination PDF path.
        generated_at: ISO timestamp string.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    if generated_at is None:
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=(PAGE_W, PAGE_H),
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title="GP Screen Failure Analysis — Campaign Summary",
        author="GP Screens Analysis",
    )

    story = []
    n = len(quant_records)

    # ------------------------------------------------------------------
    # Page 1 — Cover
    # ------------------------------------------------------------------
    story.append(Spacer(1, 1.5 * cm))
    story.append(Paragraph("GP Screen Failure Analysis", STYLE_H1))
    story.append(Paragraph("Campaign Summary Report", STYLE_H1))
    story.append(Spacer(1, 0.6 * cm))
    story.append(Paragraph(f"Total images analysed: <b>{n}</b>", STYLE_BODY))
    story.append(Paragraph(f"Generated: {generated_at}", STYLE_SMALL))
    story.append(Spacer(1, 0.5 * cm))

    # Aggregate key metrics
    all_erosion = [r.erosion_pct for r in quant_records]
    all_defects = [r.n_defects for r in quant_records]
    total_review = sum(r.n_requires_review for r in quant_records)
    avg_erosion = sum(all_erosion) / n if n else 0
    max_erosion_rec = max(quant_records, key=lambda r: r.erosion_pct) if quant_records else None

    cover_data = [
        ["Metric", "Value"],
        ["Images Analysed", str(n)],
        ["Mean Erosion %", f"{avg_erosion:.1f} %"],
        ["Max Erosion %", f"{max_erosion_rec.erosion_pct:.1f} % ({max_erosion_rec.source_filename})" if max_erosion_rec else "—"],
        ["Total Defects", str(sum(all_defects))],
        ["Flagged for Review", str(total_review)],
    ]
    cover_table = Table(cover_data, colWidths=[CONTENT_W * 0.5, CONTENT_W * 0.5])
    cover_table.setStyle(TABLE_HEADER_STYLE)
    story.append(cover_table)

    # ------------------------------------------------------------------
    # Page 2 — Summary table
    # ------------------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("Summary Table", STYLE_H2))

    sev_display = {
        "low":      '<font color="#33A633"><b>LOW</b></font>',
        "medium":   '<font color="#E09914"><b>MEDIUM</b></font>',
        "high":     '<font color="#D94814"><b>HIGH</b></font>',
        "critical": '<font color="#BF0D0D"><b>CRITICAL</b></font>',
    }

    tbl_data = [["Filename", "Erosion%", "Defects", "Severity", "Dominant Type", "Review"]]
    for r in quant_records:
        tbl_data.append([
            Paragraph(r.source_filename, STYLE_SMALL),
            f"{r.erosion_pct:.1f}",
            str(r.n_defects),
            Paragraph(sev_display.get(r.overall_severity, r.overall_severity), STYLE_SMALL),
            Paragraph(r.dominant_failure_type.replace("_", " ").title(), STYLE_SMALL),
            str(r.n_requires_review) if r.n_requires_review else "—",
        ])

    tbl = Table(
        tbl_data,
        colWidths=[
            CONTENT_W * 0.28, CONTENT_W * 0.11, CONTENT_W * 0.10,
            CONTENT_W * 0.14, CONTENT_W * 0.25, CONTENT_W * 0.12,
        ],
    )
    tbl.setStyle(SUMMARY_TABLE_STYLE)
    story.append(tbl)

    # ------------------------------------------------------------------
    # Page 3 — Charts
    # ------------------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("Analysis Charts", STYLE_H2))

    # Erosion bar chart
    erosion_chart_path = charts_dir / "erosion_bar.png"
    erosion_bar_chart(
        filenames=[r.source_filename for r in quant_records],
        erosion_pcts=[r.erosion_pct for r in quant_records],
        severities=[r.overall_severity for r in quant_records],
        output_path=erosion_chart_path,
        figsize=(8, max(3.0, n * 0.45)),
    )
    ec_img = _scale_chart(erosion_chart_path, CONTENT_W, PAGE_H * 0.38)
    if ec_img:
        story.append(ec_img)
        story.append(Paragraph("Figure 1. Erosion % per image. Dashed lines at 20% (medium threshold) and 50% (high threshold).", STYLE_CAPTION))
    story.append(Spacer(1, 0.3 * cm))

    # Aggregate failure type counts
    total_ft: dict[str, int] = {}
    for r in quant_records:
        ft_data = json.loads(r.failure_type_breakdown_json or "{}")
        for ft, stats in ft_data.items():
            total_ft[ft] = total_ft.get(ft, 0) + stats["count"]

    ft_chart_path = charts_dir / "failure_type_bar.png"
    failure_type_chart(
        type_counts=dict(sorted(total_ft.items(), key=lambda x: -x[1])),
        output_path=ft_chart_path,
        figsize=(7, max(2.5, len(total_ft) * 0.55)),
    )
    ft_img = _scale_chart(ft_chart_path, CONTENT_W * 0.62, PAGE_H * 0.32)

    # Severity distribution pie
    sev_counts: dict[str, int] = Counter(r.overall_severity for r in quant_records)
    sev_chart_path = charts_dir / "severity_pie.png"
    severity_pie_chart(dict(sev_counts), sev_chart_path, figsize=(3.5, 3.5))
    sev_img = _scale_chart(sev_chart_path, CONTENT_W * 0.35, PAGE_H * 0.32)

    if ft_img and sev_img:
        side_by_side = Table(
            [[ft_img, sev_img]],
            colWidths=[CONTENT_W * 0.62, CONTENT_W * 0.38],
        )
        story.append(side_by_side)
        story.append(Paragraph(
            "Figure 2 (left). Failure type distribution — total detections across all images.   "
            "Figure 3 (right). Image severity distribution.",
            STYLE_CAPTION,
        ))
    elif ft_img:
        story.append(ft_img)

    # ------------------------------------------------------------------
    # Page 4+ — Per-image thumbnail grid (2 per row)
    # ------------------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("Per-Image Overview", STYLE_H2))

    thumb_w = (CONTENT_W - 0.4 * cm) / 2
    thumb_h = thumb_w * (676 / 640)    # annotated aspect ratio (640×676)

    row = []
    for rec in quant_records:
        ann = annotation_map.get(rec.image_id)
        ann_path = Path(ann.annotated_path) if ann and ann.annotated_path else None
        thumb = _scale_thumbnail(ann_path, thumb_w, thumb_h)

        sev_col_hex = {
            "low": "#33A633", "medium": "#E09914",
            "high": "#D94814", "critical": "#BF0D0D",
        }.get(rec.overall_severity, "#666")

        caption = Paragraph(
            f"<b>{rec.source_filename}</b><br/>"
            f'Erosion: <b>{rec.erosion_pct:.1f}%</b>  |  '
            f'Severity: <font color="{sev_col_hex}"><b>{rec.overall_severity.upper()}</b></font><br/>'
            f"Defects: {rec.n_defects}  |  Dominant: {rec.dominant_failure_type.replace('_',' ')}",
            STYLE_SMALL,
        )

        cell = [thumb or Paragraph("Image not available", STYLE_SMALL), caption]
        row.append(cell)

        if len(row) == 2:
            grid = Table(
                [[[row[0][0], row[0][1]], [row[1][0], row[1][1]]]],
                colWidths=[thumb_w + 0.2 * cm, thumb_w + 0.2 * cm],
            )
            story.append(grid)
            story.append(Spacer(1, 0.3 * cm))
            row = []

    # Remaining odd image
    if row:
        cell = row[0]
        grid = Table(
            [[[cell[0], cell[1]], ""]],
            colWidths=[thumb_w + 0.2 * cm, thumb_w + 0.2 * cm],
        )
        story.append(grid)

    doc.build(story)
    return output_path
