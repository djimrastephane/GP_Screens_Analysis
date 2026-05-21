"""Classification pipeline: load detections + masks → classify → persist."""

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
from src.segmentation.contour_mask import ContourMaskSegmenter

from .classifier import RuleBasedClassifier
from .store import ensure_table, upsert_classification_run


@dataclass
class ClassificationSummary:
    total: int
    processed: int
    skipped: int
    total_classifications: int
    requires_review: int
    failure_type_counts: dict[str, int]
    errors: list[str]


def classify_all(
    db_path: str | Path,
    config_path: str | Path | None = None,
    detector_name: str = "ContourDetector",
) -> ClassificationSummary:
    """Classify all detected regions.

    Reuses the ContourMaskSegmenter to re-derive individual masks in-memory
    (fast, avoids storing per-detection mask files).

    Args:
        db_path: SQLite database with detection_runs and segmentation_runs.
        config_path: optional path to severity_config.yaml.
        detector_name: which detector's results to classify.
    """
    db_path = Path(db_path)
    config_path = Path(config_path) if config_path else None

    classifier = RuleBasedClassifier(config_path)
    segmenter = ContourMaskSegmenter()

    SessionFactory = init_db(db_path)
    det_ensure(db_path)
    prep_ensure(db_path)
    seg_ensure(db_path)
    ensure_table(db_path)

    summary = ClassificationSummary(
        total=0, processed=0, skipped=0,
        total_classifications=0, requires_review=0,
        failure_type_counts={}, errors=[],
    )

    with SessionFactory() as session:
        prep_map = {r.image_id: r for r in get_all_preprocessed(session)}
        seg_map = {
            r.image_id: r
            for r in get_segmentation_runs(session)
        }
        det_runs = [r for r in get_detection_runs(session) if r.detector_name == detector_name]
        summary.total = len(det_runs)

        for det_run in det_runs:
            try:
                prep_rec = prep_map.get(det_run.image_id)
                if not prep_rec or not prep_rec.preprocessed_png_path:
                    logger.warning(f"No preprocessed PNG for {det_run.source_filename}")
                    summary.skipped += 1
                    continue

                img = np.array(
                    Image.open(prep_rec.preprocessed_png_path).convert("RGB"),
                    dtype=np.uint8,
                )

                detections: list[dict] = json.loads(det_run.detections_json or "[]")

                # Re-derive individual masks in-memory
                masks: list[np.ndarray] = [
                    segmenter.segment_one(img, d["x1"], d["y1"], d["x2"], d["y2"], d["class_name"])
                    for d in detections
                ]

                # Get composite defect area from segmentation results if available
                seg_rec = seg_map.get(det_run.image_id)
                composite_area = seg_rec.composite_defect_area_pct if seg_rec else 0.0

                result = classifier.classify_image(
                    image_id=det_run.image_id,
                    source_filename=det_run.source_filename,
                    img=img,
                    detections=detections,
                    masks=masks,
                    composite_defect_area_pct=composite_area,
                    run_timestamp=datetime.now(timezone.utc).isoformat(),
                )
                upsert_classification_run(session, result)

                summary.processed += 1
                summary.total_classifications += result.count
                summary.requires_review += result.review_count

                for c in result.classifications:
                    ft = c.failure_type.value
                    summary.failure_type_counts[ft] = (
                        summary.failure_type_counts.get(ft, 0) + 1
                    )

                review_str = f" ({result.review_count} need review)" if result.review_count else ""
                logger.info(
                    f"{det_run.source_filename} — dominant={result.dominant_failure_type}, "
                    f"severity={result.overall_severity}, "
                    f"{result.count} classifications{review_str}"
                )

            except Exception as exc:
                msg = f"{det_run.source_filename}: {exc}"
                logger.error(msg)
                summary.errors.append(msg)

        session.commit()

    logger.info(
        f"Classification complete — {summary.processed}/{summary.total}, "
        f"{summary.total_classifications} total, "
        f"{summary.requires_review} require review"
    )
    return summary
