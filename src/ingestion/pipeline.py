"""Ingestion pipeline: discover → validate → load → quality → metadata → persist."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from .database import init_db, upsert_image_metadata
from .loader import discover_images, generate_image_id, load_image, validate_file
from .metadata import extract_metadata
from .quality import assess_quality


@dataclass
class IngestionSummary:
    total_discovered: int
    ingested: int
    skipped_invalid: int
    flagged: int          # processable but with quality flags
    not_processable: int  # quality so poor inference is unreliable
    errors: list[str]


def ingest_directory(
    image_dir: str | Path,
    db_path: str | Path,
    config_path: str | Path | None = None,
) -> IngestionSummary:
    """Run full ingestion for all images in image_dir, persist to SQLite at db_path.

    Args:
        image_dir: directory containing source images (read-only).
        db_path: path to SQLite database file (created if absent).
        config_path: optional path to severity_config.yaml.
    """
    image_dir = Path(image_dir)
    db_path = Path(db_path)
    config_path = Path(config_path) if config_path else None

    SessionFactory = init_db(db_path)

    paths = discover_images(image_dir)
    logger.info(f"Discovered {len(paths)} image(s) in {image_dir}")

    summary = IngestionSummary(
        total_discovered=len(paths),
        ingested=0,
        skipped_invalid=0,
        flagged=0,
        not_processable=0,
        errors=[],
    )

    with SessionFactory() as session:
        for path in paths:
            try:
                valid, reason = validate_file(path)
                if not valid:
                    logger.warning(f"Skipping {path.name}: {reason}")
                    summary.skipped_invalid += 1
                    continue

                pil_img, arr = load_image(path)
                quality = assess_quality(arr, config_path)
                image_id = generate_image_id(path)
                meta = extract_metadata(path, image_id, pil_img, quality)

                upsert_image_metadata(session, meta)
                summary.ingested += 1

                if not quality.is_processable:
                    summary.not_processable += 1
                    logger.warning(
                        f"{path.name} [{image_id}] — NOT PROCESSABLE: {quality.quality_flag}"
                    )
                elif quality.flags:
                    summary.flagged += 1
                    logger.warning(
                        f"{path.name} [{image_id}] — quality flags: {quality.quality_flag}"
                    )
                else:
                    logger.info(f"{path.name} [{image_id}] — ok")

            except Exception as exc:
                msg = f"{path.name}: {exc}"
                logger.error(msg)
                summary.errors.append(msg)

        session.commit()

    logger.info(
        f"Ingestion complete — "
        f"{summary.ingested} ingested, "
        f"{summary.skipped_invalid} skipped, "
        f"{summary.flagged} flagged, "
        f"{summary.not_processable} not processable, "
        f"{len(summary.errors)} errors"
    )
    return summary
