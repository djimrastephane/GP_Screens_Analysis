"""CLI: score the segmentation stage's composite mask against human-labeled
ground truth.

Reads polygon-labeled masks from data/annotations/*.json (see
data/annotations/README.md for the format) and compares each against the
most recent matching segmentation run's COMPOSITE mask PNG stored for that
image in images.db. Reports pooled precision/recall/Dice and mean IoU.

Only the composite mask is scored — the segmentation pipeline doesn't persist
per-instance predicted masks, only per-image summary stats plus the union of
all masks for that image (see src/segmentation/pipeline.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2


def _load_ground_truth(annotations_dir: Path) -> list[dict]:
    labeled_images = []
    for path in sorted(annotations_dir.glob("*.json")):
        with path.open() as f:
            data = json.load(f)
        if "source_filename" not in data or "masks" not in data:
            print(f"  Skipping {path.name}: missing source_filename or masks")
            continue
        labeled_images.append(data)
    return labeled_images


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation accuracy against ground-truth polygon masks."
    )
    parser.add_argument("--annotations-dir", default="data/annotations")
    parser.add_argument("--db", default="data/processed/images.db")
    parser.add_argument("--segmenter-name", default="ContourMaskSegmenter",
                        help="Which segmenter's runs to score — this is the segmenter class's "
                             "own .name (e.g. ContourMaskSegmenter, GrabCutSegmenter, SAMSegmenter), "
                             "as stored in segmentation_runs.segmenter_name, NOT the "
                             "--segmenter CLI short name used by scripts/batch_segment.py.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    annotations_dir = project_root / args.annotations_dir
    db_path = project_root / args.db

    from src.segmentation import (
        ensure_table, get_segmentation_runs, polygons_to_mask, score_masks, aggregate_mask_metrics,
    )
    from src.ingestion.database import init_db

    labeled_images = _load_ground_truth(annotations_dir)
    if not labeled_images:
        print(
            f"No ground-truth mask annotations found in {annotations_dir}.\n"
            "See data/annotations/README.md for the expected format."
        )
        return

    SessionFactory = init_db(db_path)
    ensure_table(db_path)
    with SessionFactory() as session:
        runs_by_filename = {
            r.source_filename: r
            for r in get_segmentation_runs(session)
            if r.segmenter_name == args.segmenter_name
        }

    per_image_results = []
    header = f"{'Image':<25} {'GT px':>10} {'Pred px':>10} {'IoU':>7} {'Dice':>7}"
    print(header)
    print("-" * len(header))

    for labeled in labeled_images:
        filename = labeled["source_filename"]
        polygons = [m["points"] for m in labeled["masks"]]

        run = runs_by_filename.get(filename)
        if run is None or not run.mask_png_path:
            print(f"{filename:<25} {'—':>10} {'—':>10} {'—':>7} {'—':>7} (no {args.segmenter_name} run found)")
            continue

        pred_mask = cv2.imread(run.mask_png_path, cv2.IMREAD_GRAYSCALE)
        if pred_mask is None:
            print(f"{filename:<25} {'—':>10} {'—':>10} {'—':>7} {'—':>7} (mask file missing on disk)")
            continue

        h, w = pred_mask.shape[:2]
        gt_mask = polygons_to_mask(polygons, h, w)

        result = score_masks(pred_mask, gt_mask, filename)
        per_image_results.append(result)

        print(
            f"{filename:<25} {result.gt_area_px:>10} {result.pred_area_px:>10} "
            f"{result.iou:>7.3f} {result.dice:>7.3f}"
        )

    if not per_image_results:
        print(f"\nNo labeled images had a matching {args.segmenter_name} run — nothing to score.")
        return

    metrics = aggregate_mask_metrics(per_image_results)
    print("\n--- Overall (composite mask, pooled pixels) ---")
    print(f"  Precision : {metrics.precision:.3f}")
    print(f"  Recall    : {metrics.recall:.3f}")
    print(f"  Dice      : {metrics.dice:.3f}")
    print(f"  Mean IoU  : {metrics.mean_iou:.3f} (mean of per-image IoU)")
    print(f"  Images scored : {metrics.n_images}")


if __name__ == "__main__":
    main()
