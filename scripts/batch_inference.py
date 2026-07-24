"""CLI: run the full GP screen analysis pipeline end-to-end, in one command.

Usage:
    python scripts/batch_inference.py --input Image/ --output outputs/

Chains, in order: ingest -> preprocess -> detect -> segment -> classify ->
quantify -> annotate -> report. Each stage persists its results to the
shared SQLite database before the next stage reads from it — equivalent to
running each scripts/batch_*.py script manually, in sequence, with matching
default paths.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full GP screen analysis pipeline end-to-end."
    )
    parser.add_argument("--input", default="Image", help="Source image directory (default: Image/)")
    parser.add_argument("--output", default="outputs",
                        help="Root directory for masks/overlays/panels/reports (default: outputs/)")
    parser.add_argument("--data-dir", default="data/processed",
                        help="Root directory for the SQLite DB, preprocessed images, and CSV export "
                             "(default: data/processed/)")
    parser.add_argument("--db", default=None, help="SQLite database path (default: <data-dir>/images.db)")
    parser.add_argument("--severity-config", default="configs/severity_config.yaml")
    parser.add_argument("--preprocessing-config", default="configs/preprocessing_config.yaml")
    parser.add_argument("--detector", default="contour", choices=["contour", "yolo"],
                        help="Detector type (default: contour)")
    parser.add_argument("--weights", default=None, help="YOLO weights path (only used with --detector yolo)")
    parser.add_argument("--no-color", action="store_true", help="Disable colour-anomaly contour detection")
    parser.add_argument("--segmenter", default="contour_mask", choices=["contour_mask", "grabcut", "sam"],
                        help="Segmenter type (default: contour_mask)")
    parser.add_argument("--no-panels", action="store_true", help="Skip 3-panel composite images")
    parser.add_argument("--skip-reports", action="store_true", help="Skip PDF report generation")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    image_dir = root / args.input
    output_dir = root / args.output
    data_dir = root / args.data_dir
    db_path = (root / args.db) if args.db else (data_dir / "images.db")
    severity_config = root / args.severity_config
    preprocessing_config = root / args.preprocessing_config

    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"
    panels_dir = output_dir / "panels"
    reports_dir = output_dir / "reports"

    db_path.parent.mkdir(parents=True, exist_ok=True)

    detector_name = "YOLOv8Detector" if args.detector == "yolo" else "ContourDetector"

    start = time.monotonic()
    errors: list[str] = []

    _section("1/8 Ingestion")
    from src.ingestion import ingest_directory
    summary = ingest_directory(image_dir, db_path, severity_config)
    print(f"  Discovered: {summary.total_discovered}  Ingested: {summary.ingested}  "
          f"Skipped: {summary.skipped_invalid}  Flagged: {summary.flagged}  "
          f"Not processable: {summary.not_processable}")
    errors += [f"[ingest] {e}" for e in summary.errors]

    _section("2/8 Preprocessing")
    from src.preprocessing.pipeline import preprocess_all
    summary = preprocess_all(db_path, data_dir, preprocessing_config)
    print(f"  Processed: {summary.processed}/{summary.total}  Enhanced: {summary.enhanced}  "
          f"Skipped: {summary.skipped_not_processable}")
    errors += [f"[preprocess] {e}" for e in summary.errors]

    _section("3/8 Detection")
    from src.detection import detect_all
    detector_config: dict = {"detector": args.detector}
    if args.weights:
        detector_config["weights_path"] = args.weights
    if args.no_color:
        detector_config["run_color_anomaly"] = False
    summary = detect_all(db_path, detector_config=detector_config)
    print(f"  Processed: {summary.processed}/{summary.total}  "
          f"Detections: {summary.total_detections}  Skipped: {summary.skipped_no_preprocessed}")
    errors += [f"[detect] {e}" for e in summary.errors]

    _section("4/8 Segmentation")
    from src.segmentation import segment_all
    summary = segment_all(
        db_path=db_path, mask_dir=masks_dir, overlay_dir=overlays_dir,
        segmenter_name=args.segmenter, detector_name=detector_name,
    )
    print(f"  Processed: {summary.processed}/{summary.total}  "
          f"Masks: {summary.total_masks}  Skipped: {summary.skipped}")
    errors += [f"[segment] {e}" for e in summary.errors]

    _section("5/8 Classification")
    from src.classification import classify_all
    summary = classify_all(db_path, severity_config, detector_name)
    print(f"  Processed: {summary.processed}/{summary.total}  "
          f"Classifications: {summary.total_classifications}  Needs review: {summary.requires_review}")
    errors += [f"[classify] {e}" for e in summary.errors]

    _section("6/8 Quantification")
    from src.quantification import quantify_all
    summary = quantify_all(db_path, data_dir, severity_config)
    print(f"  Processed: {summary.processed}/{summary.total}  Skipped: {summary.skipped}")
    if summary.csv_path:
        print(f"  CSV exported: {summary.csv_path}")
    errors += [f"[quantify] {e}" for e in summary.errors]

    _section("7/8 Annotation")
    from src.annotation import annotate_all
    summary = annotate_all(
        db_path=db_path, overlay_dir=overlays_dir,
        panels_dir=None if args.no_panels else panels_dir,
        include_panels=not args.no_panels,
    )
    print(f"  Processed: {summary.processed}/{summary.total}  Skipped: {summary.skipped}")
    errors += [f"[annotate] {e}" for e in summary.errors]

    if not args.skip_reports:
        _section("8/8 Reporting")
        from src.reporting import generate_all_reports
        summary = generate_all_reports(db_path, reports_dir)
        print(f"  Per-image PDFs: {summary.per_image_generated}/{summary.total_images}  "
              f"Summary PDF: {'OK' if summary.summary_generated else 'FAILED'}")
        errors += [f"[report] {e}" for e in summary.errors]
    else:
        _section("8/8 Reporting")
        print("  Skipped (--skip-reports)")

    elapsed = time.monotonic() - start
    _section("Pipeline complete")
    print(f"  Elapsed  : {elapsed:.1f}s")
    print(f"  Database : {db_path}")
    print(f"  Outputs  : {output_dir}")

    if errors:
        print(f"\n{len(errors)} error(s) across all stages:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
