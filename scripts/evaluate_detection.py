"""CLI: score detector output against human-labeled ground truth.

Reads labeled boxes from data/annotations/*.json (see
data/annotations/README.md for the format) and compares them against the
most recent detection run stored for the matching image in images.db.
Reports precision, recall, F1, and mean IoU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_ground_truth(annotations_dir: Path) -> list[dict]:
    labeled_images = []
    for path in sorted(annotations_dir.glob("*.json")):
        with path.open() as f:
            data = json.load(f)
        if "source_filename" not in data or "boxes" not in data:
            print(f"  Skipping {path.name}: missing source_filename or boxes")
            continue
        labeled_images.append(data)
    return labeled_images


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate detection accuracy against ground-truth annotations."
    )
    parser.add_argument("--annotations-dir", default="data/annotations")
    parser.add_argument("--db", default="data/processed/images.db")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    annotations_dir = project_root / args.annotations_dir
    db_path = project_root / args.db

    from src.detection import ensure_table, get_detection_runs, match_detections, aggregate_metrics
    from src.ingestion.database import init_db

    labeled_images = _load_ground_truth(annotations_dir)
    if not labeled_images:
        print(
            f"No ground-truth annotation files found in {annotations_dir}.\n"
            "See data/annotations/README.md for the expected format."
        )
        return

    SessionFactory = init_db(db_path)
    ensure_table(db_path)
    with SessionFactory() as session:
        runs_by_filename = {r.source_filename: r for r in get_detection_runs(session)}

    per_image_results = []
    header = f"{'Image':<25} {'GT':>4} {'Pred':>5} {'TP':>4} {'FP':>4} {'FN':>4}"
    print(header)
    print("-" * len(header))

    for labeled in labeled_images:
        filename = labeled["source_filename"]
        gt_boxes = [SimpleNamespace(**b) for b in labeled["boxes"]]

        run = runs_by_filename.get(filename)
        if run is None:
            print(f"{filename:<25} {len(gt_boxes):>4}     —    —    —    — (no detection run found)")
            continue

        detections = json.loads(run.detections_json or "[]")
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        pred_boxes = [SimpleNamespace(**d) for d in detections]

        result = match_detections(pred_boxes, gt_boxes, iou_threshold=args.iou_threshold)
        per_image_results.append(result)

        print(
            f"{filename:<25} {len(gt_boxes):>4} {len(pred_boxes):>5} "
            f"{result.true_positives:>4} {result.false_positives:>4} {result.false_negatives:>4}"
        )

    if not per_image_results:
        print("\nNo labeled images had a matching detection run — nothing to score.")
        return

    metrics = aggregate_metrics(per_image_results)
    print("\n--- Overall ---")
    print(f"  Precision : {metrics.precision:.3f}")
    print(f"  Recall    : {metrics.recall:.3f}")
    print(f"  F1        : {metrics.f1:.3f}")
    print(f"  Mean IoU  : {metrics.mean_iou:.3f} (matched boxes only)")
    print(f"  TP / FP / FN : {metrics.true_positives} / {metrics.false_positives} / {metrics.false_negatives}")


if __name__ == "__main__":
    main()
