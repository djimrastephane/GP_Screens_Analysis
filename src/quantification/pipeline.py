"""Quantification pipeline: join all pipeline stages → compute measurements → export."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from src.ingestion.database import init_db
from src.preprocessing.store import ensure_table as prep_ensure, get_all_preprocessed
from src.segmentation.store import ensure_table as seg_ensure, get_segmentation_runs
from src.classification.store import ensure_table as cls_ensure, get_classification_runs

from .aggregator import aggregate
from .measurements import load_scale, screen_region_area_px
from .report import QuantificationReport, build_report, export_csv
from .store import ensure_table, get_all_reports, upsert_report


@dataclass
class QuantificationSummary:
    total: int
    processed: int
    skipped: int
    csv_path: str | None
    errors: list[str]


def quantify_all(
    db_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
) -> QuantificationSummary:
    """Join all pipeline stages and produce per-image QuantificationReports.

    Args:
        db_path: SQLite database with all pipeline tables.
        output_dir: root output directory for CSV export.
        config_path: optional path to severity_config.yaml (for scale and severity thresholds).
    """
    db_path = Path(db_path)
    output_dir = Path(output_dir)
    config_path = Path(config_path) if config_path else None

    pixels_per_mm = load_scale(config_path)
    if pixels_per_mm:
        logger.info(f"Scale calibration: {pixels_per_mm:.4f} px/mm")
    else:
        logger.info("No scale calibration — measurements will be pixel-only")

    SessionFactory = init_db(db_path)
    prep_ensure(db_path)
    seg_ensure(db_path)
    cls_ensure(db_path)
    ensure_table(db_path)

    summary = QuantificationSummary(
        total=0, processed=0, skipped=0, csv_path=None, errors=[]
    )

    reports: list[QuantificationReport] = []

    with SessionFactory() as session:
        prep_map = {r.image_id: r for r in get_all_preprocessed(session)}
        seg_map = {r.image_id: r for r in get_segmentation_runs(session)}
        cls_map = {r.image_id: r for r in get_classification_runs(session)}

        image_ids = sorted(prep_map.keys())
        summary.total = len(image_ids)

        for image_id in image_ids:
            try:
                prep_rec = prep_map.get(image_id)
                seg_rec = seg_map.get(image_id)
                cls_rec = cls_map.get(image_id)

                if not prep_rec or not seg_rec or not cls_rec:
                    logger.warning(
                        f"{image_id}: missing "
                        + (", ".join(
                            name for name, rec in [("prep", prep_rec), ("seg", seg_rec), ("cls", cls_rec)]
                            if rec is None
                        ))
                        + " — skipping"
                    )
                    summary.skipped += 1
                    continue

                screen_area = screen_region_area_px(
                    prep_rec.target_width, prep_rec.target_height,
                    prep_rec.pad_top, prep_rec.pad_bottom,
                    prep_rec.pad_left, prep_rec.pad_right,
                )

                classifications: list[dict] = json.loads(cls_rec.classifications_json or "[]")
                seg_masks: list[dict] = json.loads(seg_rec.per_mask_json or "[]")

                # Align: classification and segmentation lists are parallel by detection_index
                n = min(len(classifications), len(seg_masks))
                classifications = classifications[:n]
                seg_masks = seg_masks[:n]

                metrics = aggregate(
                    classifications=classifications,
                    seg_masks=seg_masks,
                    composite_defect_area_px=seg_rec.composite_defect_area_px,
                    screen_area_px=screen_area,
                    overall_severity=cls_rec.overall_severity,
                    dominant_failure_type=cls_rec.dominant_failure_type,
                )

                report = build_report(
                    image_id=image_id,
                    source_filename=prep_rec.source_filename,
                    image_width=prep_rec.target_width,
                    image_height=prep_rec.target_height,
                    screen_area_px=screen_area,
                    classifications=classifications,
                    seg_masks=seg_masks,
                    metrics=metrics,
                    pixels_per_mm=pixels_per_mm,
                    run_timestamp=datetime.now(timezone.utc).isoformat(),
                )
                upsert_report(session, report)
                reports.append(report)

                summary.processed += 1

                ft_summary = ", ".join(
                    f"{ft}×{s['count']}"
                    for ft, s in report.failure_type_breakdown.items()
                )
                logger.info(
                    f"{prep_rec.source_filename} — erosion={report.erosion_pct:.1f}%, "
                    f"screen_area={screen_area}px, "
                    f"defects={report.n_defects} [{ft_summary}]"
                )

            except Exception as exc:
                msg = f"{image_id}: {exc}"
                logger.error(msg)
                summary.errors.append(msg)

        session.commit()

    # Export summary CSV
    if reports:
        csv_path = output_dir / "quantification_summary.csv"
        export_csv(reports, csv_path)
        summary.csv_path = str(csv_path)
        logger.info(f"Summary CSV written: {csv_path}")

    logger.info(
        f"Quantification complete — {summary.processed}/{summary.total} processed, "
        f"{summary.skipped} skipped, {len(summary.errors)} errors"
    )
    return summary
