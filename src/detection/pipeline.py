"""Detection pipeline: load preprocessed PNGs → run detector → persist results."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image

from src.ingestion.database import init_db
from src.preprocessing.store import get_all_preprocessed, ensure_table as ensure_prep_table

from .base import BaseDetector, DetectionResult
from .contour import ContourDetector, ContourConfig
from .store import ensure_table, upsert_detection_run


@dataclass
class DetectionSummary:
    total: int
    processed: int
    skipped_no_preprocessed: int
    total_detections: int
    errors: list[str]


def _load_detector(config: dict) -> BaseDetector:
    """Build a detector from a config dict (from YAML or defaults)."""
    detector_type = config.get("detector", "contour")

    if detector_type == "yolo":
        from .yolo import YOLOv8Detector
        return YOLOv8Detector(
            weights_path=config.get("weights_path"),
            conf_threshold=config.get("conf_threshold", 0.25),
            iou_threshold=config.get("iou_threshold", 0.45),
            device=config.get("device", "cpu"),
        )

    # Default: contour detector — values fall back to ContourConfig dataclass defaults
    cfg = ContourConfig(
        close_kernel_size=config.get("close_kernel_size", 21),
        open_kernel_size=config.get("open_kernel_size", 5),
        adaptive_block_size=config.get("adaptive_block_size", 51),
        adaptive_c=config.get("adaptive_c", 6),
        min_area_frac=config.get("min_area_frac", 0.001),
        max_area_frac=config.get("max_area_frac", 0.40),
        min_aspect=config.get("min_aspect", 0.12),
        max_aspect=config.get("max_aspect", 7.0),
        nms_iou=config.get("nms_iou", 0.35),
        cross_iou=config.get("cross_iou", 0.50),
        containment_threshold=config.get("containment_threshold", 0.80),
        run_color_anomaly=config.get("run_color_anomaly", True),
    )
    return ContourDetector(cfg)


def detect_all(
    db_path: str | Path,
    detector: BaseDetector | None = None,
    detector_config: dict | None = None,
) -> DetectionSummary:
    """Run detection on all preprocessed images.

    Loads uint8 PNGs from preprocessed_png_path in the DB. Results are
    persisted to the detection_runs table.

    Args:
        db_path: path to existing SQLite database (must have preprocessed_images table).
        detector: optional pre-built detector instance. If None, builds one
                  from detector_config or uses ContourDetector defaults.
        detector_config: dict passed to _load_detector if detector is None.
    """
    db_path = Path(db_path)

    if detector is None:
        detector = _load_detector(detector_config or {})

    if not detector.ready:
        logger.warning(
            f"{detector.name} is not ready (missing weights?). "
            "No detections will be produced."
        )

    SessionFactory = init_db(db_path)
    ensure_prep_table(db_path)
    ensure_table(db_path)

    summary = DetectionSummary(
        total=0, processed=0, skipped_no_preprocessed=0,
        total_detections=0, errors=[]
    )

    with SessionFactory() as session:
        prep_records = get_all_preprocessed(session)
        summary.total = len(prep_records)

        for rec in prep_records:
            try:
                png_path = rec.preprocessed_png_path
                if not png_path or not Path(png_path).exists():
                    logger.warning(
                        f"No preprocessed PNG for {rec.source_filename}, skipping"
                    )
                    summary.skipped_no_preprocessed += 1
                    continue

                img = np.array(Image.open(png_path).convert("RGB"), dtype=np.uint8)
                h, w = img.shape[:2]

                detections = detector.detect(img) if detector.ready else []

                result = DetectionResult(
                    image_id=rec.image_id,
                    source_filename=rec.source_filename,
                    detector_name=detector.name,
                    model_version=detector.version,
                    detections=detections,
                    image_width=w,
                    image_height=h,
                    run_timestamp=datetime.now(timezone.utc).isoformat(),
                )
                upsert_detection_run(session, result)

                summary.processed += 1
                summary.total_detections += result.count

                logger.info(
                    f"{rec.source_filename} — {result.count} detection(s)"
                    + (
                        f" [max_conf={result.max_confidence:.3f}]"
                        if result.count
                        else ""
                    )
                )

            except Exception as exc:
                msg = f"{rec.source_filename}: {exc}"
                logger.error(msg)
                summary.errors.append(msg)

        session.commit()

    logger.info(
        f"Detection complete — {summary.processed}/{summary.total} processed, "
        f"{summary.total_detections} total detections, "
        f"{len(summary.errors)} errors"
    )
    return summary
