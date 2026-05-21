"""Preprocessing pipeline: load → enhance → letterbox → normalize → save."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from loguru import logger
from PIL import Image

from src.ingestion.database import get_all_images, init_db
from src.ingestion.loader import load_image

from .enhance import apply_clahe, apply_denoise, select_enhancements
from .normalize import to_float32, to_imagenet, to_uint8
from .resize import PaddingInfo, letterbox
from .store import ensure_table, upsert_preprocessed


@dataclass
class PreprocessingConfig:
    target_width: int = 640
    target_height: int = 640
    letterbox_fill: int = 114
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: tuple[int, int] = (8, 8)
    denoise_h: float = 10.0
    denoise_template_window: int = 7
    denoise_search_window: int = 21
    normalisation: str = "float32"  # "float32" | "imagenet"
    save_png: bool = True
    save_npy: bool = True


def load_config(config_path: Path | None) -> PreprocessingConfig:
    if config_path is None or not config_path.exists():
        return PreprocessingConfig()
    with config_path.open() as f:
        raw = yaml.safe_load(f)
    cfg = PreprocessingConfig()
    if "target_size" in raw:
        cfg.target_width = raw["target_size"].get("width", cfg.target_width)
        cfg.target_height = raw["target_size"].get("height", cfg.target_height)
    cfg.letterbox_fill = raw.get("letterbox_fill", cfg.letterbox_fill)
    if "clahe" in raw:
        cfg.clahe_clip_limit = raw["clahe"].get("clip_limit", cfg.clahe_clip_limit)
        tg = raw["clahe"].get("tile_grid_size", list(cfg.clahe_tile_grid))
        cfg.clahe_tile_grid = tuple(tg)
    if "denoise" in raw:
        cfg.denoise_h = raw["denoise"].get("h", cfg.denoise_h)
        cfg.denoise_template_window = raw["denoise"].get(
            "template_window", cfg.denoise_template_window
        )
        cfg.denoise_search_window = raw["denoise"].get(
            "search_window", cfg.denoise_search_window
        )
    cfg.normalisation = raw.get("normalisation", cfg.normalisation)
    cfg.save_png = raw.get("save_png", cfg.save_png)
    cfg.save_npy = raw.get("save_npy", cfg.save_npy)
    return cfg


def preprocess_image(
    img: np.ndarray,
    quality_flags: list[str],
    cfg: PreprocessingConfig,
) -> tuple[np.ndarray, np.ndarray, PaddingInfo, list[str]]:
    """Run the full preprocessing chain for one image.

    Returns:
        preprocessed_float: float32 normalized array (model input)
        preprocessed_uint8: uint8 version for saving/display
        padding_info: letterbox geometry for coordinate remapping
        enhancements: list of enhancement names actually applied
    """
    enhanced = img.copy()
    enhancements = select_enhancements(quality_flags)

    if "denoise" in enhancements:
        enhanced = apply_denoise(
            enhanced,
            h=cfg.denoise_h,
            template_window=cfg.denoise_template_window,
            search_window=cfg.denoise_search_window,
        )
    if "clahe" in enhancements:
        enhanced = apply_clahe(
            enhanced,
            clip_limit=cfg.clahe_clip_limit,
            tile_grid_size=cfg.clahe_tile_grid,
        )

    padded, padding_info = letterbox(
        enhanced,
        target_width=cfg.target_width,
        target_height=cfg.target_height,
        fill=cfg.letterbox_fill,
    )

    if cfg.normalisation == "imagenet":
        preprocessed_float = to_imagenet(padded)
    else:
        preprocessed_float = to_float32(padded)

    preprocessed_uint8 = to_uint8(
        preprocessed_float if cfg.normalisation == "float32"
        else preprocessed_float * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    )

    return preprocessed_float, preprocessed_uint8, padding_info, enhancements


@dataclass
class PreprocessingSummary:
    total: int
    processed: int
    skipped_not_processable: int
    enhanced: int
    errors: list[str]


def preprocess_all(
    db_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
) -> PreprocessingSummary:
    """Preprocess all processable images from the ingestion DB.

    Saves outputs to output_dir/preprocessed/{png,npy}/ and records results
    in the preprocessed_images table.

    Args:
        db_path: path to existing ingestion SQLite database.
        output_dir: root output directory (e.g. data/processed).
        config_path: optional path to preprocessing_config.yaml.
    """
    db_path = Path(db_path)
    output_dir = Path(output_dir)
    config_path = Path(config_path) if config_path else None

    cfg = load_config(config_path)

    png_dir = output_dir / "preprocessed" / "png"
    npy_dir = output_dir / "preprocessed" / "npy"
    png_dir.mkdir(parents=True, exist_ok=True)
    npy_dir.mkdir(parents=True, exist_ok=True)

    SessionFactory = init_db(db_path)
    ensure_table(db_path)  # creates preprocessed_images table if missing

    summary = PreprocessingSummary(
        total=0, processed=0, skipped_not_processable=0, enhanced=0, errors=[]
    )

    with SessionFactory() as session:
        records = get_all_images(session)
        summary.total = len(records)

        for rec in records:
            try:
                if not rec.is_processable:
                    logger.warning(
                        f"Skipping {rec.source_filename} — marked not processable"
                    )
                    summary.skipped_not_processable += 1
                    continue

                pil_img, arr = load_image(Path(rec.source_path))
                quality_flags = (
                    [f.strip() for f in rec.quality_flag.split(",")]
                    if rec.quality_flag != "ok"
                    else []
                )

                float_arr, uint8_arr, pad_info, enhancements = preprocess_image(
                    arr, quality_flags, cfg
                )

                png_path = npy_path = None
                if cfg.save_png:
                    png_path = png_dir / f"{rec.image_id}.png"
                    Image.fromarray(uint8_arr).save(png_path)

                if cfg.save_npy:
                    npy_path = npy_dir / f"{rec.image_id}.npy"
                    np.save(npy_path, float_arr)

                enh_str = ", ".join(enhancements) if enhancements else "none"
                upsert_preprocessed(
                    session,
                    {
                        "image_id": rec.image_id,
                        "source_filename": rec.source_filename,
                        "preprocessed_png_path": str(png_path) if png_path else None,
                        "preprocessed_npy_path": str(npy_path) if npy_path else None,
                        "original_width": pad_info.original_width,
                        "original_height": pad_info.original_height,
                        "target_width": pad_info.target_width,
                        "target_height": pad_info.target_height,
                        "scale": round(pad_info.scale, 6),
                        "pad_top": pad_info.pad_top,
                        "pad_bottom": pad_info.pad_bottom,
                        "pad_left": pad_info.pad_left,
                        "pad_right": pad_info.pad_right,
                        "enhancements_applied": enh_str,
                        "normalisation": cfg.normalisation,
                        "preprocessed_at": datetime.now(timezone.utc).isoformat(),
                    },
                )

                summary.processed += 1
                if enhancements:
                    summary.enhanced += 1

                logger.info(
                    f"{rec.source_filename} → {pad_info.original_width}x{pad_info.original_height}"
                    f" → {pad_info.target_width}x{pad_info.target_height}"
                    f" scale={pad_info.scale:.3f}"
                    f" enhancements=[{enh_str}]"
                )

            except Exception as exc:
                msg = f"{rec.source_filename}: {exc}"
                logger.error(msg)
                summary.errors.append(msg)

        session.commit()

    logger.info(
        f"Preprocessing complete — {summary.processed}/{summary.total} processed, "
        f"{summary.enhanced} enhanced, "
        f"{summary.skipped_not_processable} skipped, "
        f"{len(summary.errors)} errors"
    )
    return summary
