"""CLI: run detection on all preprocessed GP screen images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect damage regions in GP screen images.")
    parser.add_argument("--db", default="data/processed/images.db")
    parser.add_argument(
        "--detector", default="contour", choices=["contour", "yolo"],
        help="Detector type"
    )
    parser.add_argument("--weights", default=None, help="YOLO weights path")
    parser.add_argument("--min-area", type=float, default=0.002, help="Min blob area fraction")
    parser.add_argument("--max-area", type=float, default=0.20, help="Max blob area fraction")
    parser.add_argument("--no-color", action="store_true", help="Disable colour-anomaly detector")
    parser.add_argument("--report", action="store_true", help="Print per-image detection summary")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    db_path = project_root / args.db

    from src.detection import detect_all, get_detection_runs, ensure_table
    from src.ingestion.database import init_db

    detector_config = {
        "detector": args.detector,
        "weights_path": args.weights,
        "min_area_frac": args.min_area,
        "max_area_frac": args.max_area,
        "run_color_anomaly": not args.no_color,
    }

    summary = detect_all(db_path, detector_config=detector_config)

    print("\n--- Detection Summary ---")
    print(f"  Images processed : {summary.processed}/{summary.total}")
    print(f"  Total detections : {summary.total_detections}")
    print(f"  Skipped          : {summary.skipped_no_preprocessed}")
    if summary.errors:
        print(f"  Errors           : {len(summary.errors)}")
        for e in summary.errors:
            print(f"    - {e}")

    if args.report:
        SessionFactory = init_db(db_path)
        ensure_table(db_path)
        with SessionFactory() as session:
            runs = get_detection_runs(session)
        print("\n--- Per-Image Detections ---")
        header = f"{'Filename':<25} {'Detector':<20} {'N':>4}  {'MaxConf':>8}  Classes"
        print(header)
        print("-" * 75)
        for r in runs:
            dets = json.loads(r.detections_json or "[]")
            classes = ", ".join(sorted({d["class_name"] for d in dets})) or "—"
            conf_str = f"{r.max_confidence:.3f}" if r.detection_count else "—"
            print(
                f"{r.source_filename:<25} {r.detector_name:<20} {r.detection_count:>4}"
                f"  {conf_str:>8}  {classes}"
            )


if __name__ == "__main__":
    main()
