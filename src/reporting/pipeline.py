"""Reporting pipeline: generate per-image PDFs + campaign summary PDF."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from src.ingestion.database import init_db
from src.quantification.store import ensure_table as q_ensure, get_all_reports
from src.annotation.store import ensure_table as ann_ensure, get_all_annotations

from .per_image import build_per_image_report
from .summary import build_summary_report
from .store import ensure_table, insert_report


@dataclass
class ReportingSummary:
    total_images: int
    per_image_generated: int
    summary_generated: bool
    report_dir: str
    errors: list[str]


def generate_all_reports(
    db_path: str | Path,
    reports_dir: str | Path,
) -> ReportingSummary:
    """Generate per-image PDFs and one campaign summary PDF.

    Args:
        db_path: SQLite database with all pipeline tables.
        reports_dir: output directory for PDF files.
    """
    db_path = Path(db_path)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = reports_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    now_str = datetime.now(timezone.utc).isoformat()

    SessionFactory = init_db(db_path)
    q_ensure(db_path)
    ann_ensure(db_path)
    ensure_table(db_path)

    summary = ReportingSummary(
        total_images=0,
        per_image_generated=0,
        summary_generated=False,
        report_dir=str(reports_dir),
        errors=[],
    )

    with SessionFactory() as session:
        quant_records = get_all_reports(session)
        ann_map = {r.image_id: r for r in get_all_annotations(session)}
        summary.total_images = len(quant_records)

        # Per-image reports
        for qrec in quant_records:
            try:
                ann_rec = ann_map.get(qrec.image_id)
                stem = qrec.image_id
                pdf_path = reports_dir / f"{stem}_report.pdf"

                build_per_image_report(
                    quant_record=qrec,
                    annotation_record=ann_rec,
                    output_path=pdf_path,
                    generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                )
                insert_report(
                    session, "per_image", str(pdf_path), now_str,
                    image_id=qrec.image_id, source_filename=qrec.source_filename,
                )
                summary.per_image_generated += 1
                logger.info(f"Per-image report: {pdf_path.name}")

            except Exception as exc:
                msg = f"{qrec.source_filename}: {exc}"
                logger.error(msg)
                summary.errors.append(msg)

        # Campaign summary report
        try:
            summary_path = reports_dir / "campaign_summary.pdf"
            build_summary_report(
                quant_records=quant_records,
                annotation_map=ann_map,
                charts_dir=charts_dir,
                output_path=summary_path,
                generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            )
            insert_report(session, "summary", str(summary_path), now_str)
            summary.summary_generated = True
            logger.info(f"Campaign summary: {summary_path.name}")

        except Exception as exc:
            msg = f"summary report: {exc}"
            logger.error(msg)
            summary.errors.append(msg)

        session.commit()

    logger.info(
        f"Reporting complete — {summary.per_image_generated}/{summary.total_images} "
        f"per-image PDFs, summary={'OK' if summary.summary_generated else 'FAILED'}, "
        f"{len(summary.errors)} errors"
    )
    return summary
