"""Segmentation pipeline: load detections → segment each bbox → save masks + overlays."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image

from src.ingestion.database import init_db
from src.detection.store import ensure_table as ensure_det_table, get_detection_runs

from .base import BaseSegmenter, SegmentationMask, SegmentationResult
from .contour_mask import ContourMaskSegmenter
from .grabcut import GrabCutSegmenter
from .metrics import fill_ratio, mask_area_pct, mask_area_px, union_mask, bbox_area_px
from .overlay import draw_overlay
from .store import ensure_table, upsert_segmentation_run


@dataclass
class SegmentationSummary:
    total: int
    processed: int
    skipped: int
    total_masks: int
    errors: list[str]


def _build_segmenter(name: str) -> BaseSegmenter:
    if name == "grabcut":
        return GrabCutSegmenter()
    if name == "sam":
        from .sam import SAMSegmenter
        return SAMSegmenter()
    return ContourMaskSegmenter()


def segment_all(
    db_path: str | Path,
    mask_dir: str | Path,
    overlay_dir: str | Path,
    segmenter: BaseSegmenter | None = None,
    segmenter_name: str = "contour_mask",
    detector_name: str = "ContourDetector",
) -> SegmentationSummary:
    """Segment all detections from one detector run.

    Args:
        db_path: SQLite database path (must have detection_runs table).
        mask_dir: directory to save composite mask PNGs.
        overlay_dir: directory to save annotated overlay PNGs.
        segmenter: pre-built segmenter; if None, built from segmenter_name.
        segmenter_name: "contour_mask" | "grabcut" | "sam"
        detector_name: which detector's results to segment.
    """
    db_path = Path(db_path)
    mask_dir = Path(mask_dir)
    overlay_dir = Path(overlay_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    if segmenter is None:
        segmenter = _build_segmenter(segmenter_name)

    if not segmenter.ready:
        logger.warning(f"{segmenter.name} not ready — falling back to ContourMaskSegmenter")
        segmenter = ContourMaskSegmenter()

    SessionFactory = init_db(db_path)
    ensure_det_table(db_path)
    ensure_table(db_path)

    summary = SegmentationSummary(
        total=0, processed=0, skipped=0, total_masks=0, errors=[]
    )

    with SessionFactory() as session:
        det_runs = get_detection_runs(session)
        det_runs = [r for r in det_runs if r.detector_name == detector_name]
        summary.total = len(det_runs)

        for det_run in det_runs:
            try:
                png_path = _resolve_preprocessed_png(session, det_run.image_id, db_path)
                if not png_path:
                    summary.skipped += 1
                    logger.warning(f"No preprocessed PNG for {det_run.source_filename}")
                    continue

                img = np.array(Image.open(png_path).convert("RGB"), dtype=np.uint8)
                h, w = img.shape[:2]

                detections: list[dict] = json.loads(det_run.detections_json or "[]")

                seg_masks: list[SegmentationMask] = []
                mask_arrays: list[np.ndarray] = []

                for idx, det in enumerate(detections):
                    raw_mask = segmenter.segment_one(
                        img,
                        det["x1"], det["y1"], det["x2"], det["y2"],
                        det["class_name"],
                    )
                    area_px = mask_area_px(raw_mask)
                    b_area = bbox_area_px(det["x1"], det["y1"], det["x2"], det["y2"])

                    sm = SegmentationMask(
                        detection_index=idx,
                        class_id=det["class_id"],
                        class_name=det["class_name"],
                        detection_confidence=det["confidence"],
                        source=segmenter.name,
                        mask=raw_mask,
                        defect_area_px=area_px,
                        defect_area_pct=round(mask_area_pct(raw_mask), 4),
                        bbox_area_px=b_area,
                        fill_ratio=round(fill_ratio(raw_mask, det["x1"], det["y1"], det["x2"], det["y2"]), 4),
                    )
                    seg_masks.append(sm)
                    if area_px > 0:
                        mask_arrays.append(raw_mask)

                # Composite mask — union of all individual masks
                composite = union_mask(mask_arrays) if mask_arrays else np.zeros((h, w), dtype=np.uint8)
                comp_area_px = mask_area_px(composite)
                comp_area_pct = mask_area_pct(composite)

                # Save composite mask
                mask_path = mask_dir / f"{det_run.image_id}_mask.png"
                Image.fromarray(composite).save(mask_path)

                # Draw and save overlay
                pairs = [(sm.mask, det) for sm, det in zip(seg_masks, detections)]
                overlay_img = draw_overlay(img, pairs)
                overlay_path = overlay_dir / f"{det_run.image_id}_overlay.png"
                Image.fromarray(overlay_img).save(overlay_path)

                result = SegmentationResult(
                    image_id=det_run.image_id,
                    source_filename=det_run.source_filename,
                    segmenter_name=segmenter.name,
                    model_version=segmenter.version,
                    masks=seg_masks,
                    image_width=w,
                    image_height=h,
                    run_timestamp=datetime.now(timezone.utc).isoformat(),
                    composite_defect_area_px=comp_area_px,
                    composite_defect_area_pct=round(comp_area_pct, 4),
                )
                upsert_segmentation_run(session, result, mask_path, overlay_path)

                summary.processed += 1
                summary.total_masks += len(seg_masks)

                logger.info(
                    f"{det_run.source_filename} — {len(seg_masks)} mask(s), "
                    f"composite defect area {comp_area_pct:.2f}%"
                )

            except Exception as exc:
                msg = f"{det_run.source_filename}: {exc}"
                logger.error(msg)
                summary.errors.append(msg)

        session.commit()

    logger.info(
        f"Segmentation complete — {summary.processed}/{summary.total} processed, "
        f"{summary.total_masks} masks, {len(summary.errors)} errors"
    )
    return summary


def _resolve_preprocessed_png(session, image_id: str, db_path: Path) -> Path | None:
    """Look up preprocessed PNG path from the preprocessed_images table."""
    from src.preprocessing.store import get_all_preprocessed, ensure_table as prep_ensure

    prep_ensure(db_path)
    records = get_all_preprocessed(session)
    for r in records:
        if r.image_id == image_id and r.preprocessed_png_path:
            p = Path(r.preprocessed_png_path)
            if p.exists():
                return p
    return None
