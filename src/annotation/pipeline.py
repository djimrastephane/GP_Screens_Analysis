"""Annotation pipeline: load all pipeline results → compose annotated images → save."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image

from src.ingestion.database import init_db
from src.detection.store import ensure_table as det_ensure, get_detection_runs
from src.preprocessing.store import ensure_table as prep_ensure, get_all_preprocessed
from src.segmentation.store import ensure_table as seg_ensure, get_segmentation_runs
from src.quantification.store import ensure_table as q_ensure, get_all_reports
from src.segmentation.contour_mask import ContourMaskSegmenter

from .composer import compose_annotated, compose_panels
from .draw import draw_header_bar
from .store import ensure_table, get_all_annotations, upsert_annotation


@dataclass
class AnnotationSummary:
    total: int
    processed: int
    skipped: int
    errors: list[str]


def annotate_all(
    db_path: str | Path,
    overlay_dir: str | Path,
    panels_dir: str | Path | None = None,
    include_panels: bool = True,
) -> AnnotationSummary:
    """Produce annotated overlay images (and optionally 3-panel composites).

    Reads detections, segmentation masks, quantification defects from DB.
    Re-derives individual masks in-memory via ContourMaskSegmenter.

    Args:
        db_path: SQLite database with all pipeline tables.
        overlay_dir: directory to save single annotated images.
        panels_dir: directory to save 3-panel images (optional).
        include_panels: if True, also produce side-by-side original/annotated/mask.
    """
    db_path = Path(db_path)
    overlay_dir = Path(overlay_dir)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    if panels_dir is not None and include_panels:
        panels_dir = Path(panels_dir)
        panels_dir.mkdir(parents=True, exist_ok=True)
    else:
        panels_dir = None

    segmenter = ContourMaskSegmenter()

    SessionFactory = init_db(db_path)
    det_ensure(db_path)
    prep_ensure(db_path)
    seg_ensure(db_path)
    q_ensure(db_path)
    ensure_table(db_path)

    summary = AnnotationSummary(total=0, processed=0, skipped=0, errors=[])

    with SessionFactory() as session:
        prep_map = {r.image_id: r for r in get_all_preprocessed(session)}
        det_map = {r.image_id: r for r in get_detection_runs(session)}
        seg_map = {r.image_id: r for r in get_segmentation_runs(session)}
        q_reports = {r.image_id: r for r in get_all_reports(session)}

        image_ids = sorted(prep_map.keys())
        summary.total = len(image_ids)

        for image_id in image_ids:
            try:
                prep_rec = prep_map.get(image_id)
                det_rec = det_map.get(image_id)
                q_rec = q_reports.get(image_id)

                if not prep_rec or not det_rec or not q_rec:
                    logger.warning(f"{image_id}: incomplete pipeline data — skipping")
                    summary.skipped += 1
                    continue

                if not prep_rec.preprocessed_png_path:
                    logger.warning(f"{image_id}: no preprocessed PNG — skipping")
                    summary.skipped += 1
                    continue

                img = np.array(
                    Image.open(prep_rec.preprocessed_png_path).convert("RGB"),
                    dtype=np.uint8,
                )

                detections: list[dict] = json.loads(det_rec.detections_json or "[]")
                defects: list[dict] = json.loads(q_rec.defects_json or "[]")

                n = min(len(detections), len(defects))
                detections, defects = detections[:n], defects[:n]

                # Re-derive individual masks in-memory
                masks: list[np.ndarray] = [
                    segmenter.segment_one(
                        img, d["x1"], d["y1"], d["x2"], d["y2"], d["class_name"]
                    )
                    for d in detections
                ]

                annotated = compose_annotated(
                    img=img,
                    detections=detections,
                    defects=defects,
                    masks=masks,
                    source_filename=prep_rec.source_filename,
                    erosion_pct=q_rec.erosion_pct,
                    overall_severity=q_rec.overall_severity,
                    n_requires_review=q_rec.n_requires_review,
                )

                annotated_path = overlay_dir / f"{image_id}_annotated.png"
                Image.fromarray(annotated).save(annotated_path)

                panel_path = None
                if panels_dir is not None:
                    seg_rec = seg_map.get(image_id)
                    if seg_rec and seg_rec.mask_png_path:
                        composite_mask = np.array(
                            Image.open(seg_rec.mask_png_path).convert("L"),
                            dtype=np.uint8,
                        )
                    else:
                        # Build composite mask from individual masks
                        composite_mask = (
                            np.maximum.reduce(masks) if masks
                            else np.zeros(img.shape[:2], dtype=np.uint8)
                        )

                    # Header for panel (spans 3× image width)
                    header = draw_header_bar(
                        prep_rec.source_filename, q_rec.erosion_pct,
                        q_rec.n_defects, q_rec.overall_severity,
                        q_rec.n_requires_review, img.shape[1] * 3,
                    )
                    panel = compose_panels(img, annotated[36:, :], composite_mask, header)
                    panel_path = panels_dir / f"{image_id}_panel.png"
                    Image.fromarray(panel).save(panel_path)

                upsert_annotation(
                    session, image_id, prep_rec.source_filename,
                    str(annotated_path),
                    str(panel_path) if panel_path else None,
                    datetime.now(timezone.utc).isoformat(),
                )

                summary.processed += 1
                logger.info(
                    f"{prep_rec.source_filename} → {annotated_path.name}"
                    + (f" + {panel_path.name}" if panel_path else "")
                )

            except Exception as exc:
                msg = f"{image_id}: {exc}"
                logger.error(msg)
                summary.errors.append(msg)

        session.commit()

    logger.info(
        f"Annotation complete — {summary.processed}/{summary.total} processed, "
        f"{summary.skipped} skipped, {len(summary.errors)} errors"
    )
    return summary
