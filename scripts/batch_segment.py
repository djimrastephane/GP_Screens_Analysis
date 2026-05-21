"""CLI: run segmentation on all detected regions and save masks + overlays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment detected GP screen damage regions.")
    parser.add_argument("--db", default="data/processed/images.db")
    parser.add_argument(
        "--segmenter", default="contour_mask",
        choices=["contour_mask", "grabcut", "sam"],
    )
    parser.add_argument("--detector", default="ContourDetector",
                        help="Which detector's results to segment")
    parser.add_argument("--masks-dir", default="outputs/masks")
    parser.add_argument("--overlays-dir", default="outputs/overlays")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    db_path = root / args.db
    mask_dir = root / args.masks_dir
    overlay_dir = root / args.overlays_dir

    from src.segmentation import segment_all, get_segmentation_runs, ensure_table
    from src.ingestion.database import init_db

    summary = segment_all(
        db_path=db_path,
        mask_dir=mask_dir,
        overlay_dir=overlay_dir,
        segmenter_name=args.segmenter,
        detector_name=args.detector,
    )

    print("\n--- Segmentation Summary ---")
    print(f"  Images processed : {summary.processed}/{summary.total}")
    print(f"  Total masks      : {summary.total_masks}")
    print(f"  Skipped          : {summary.skipped}")
    if summary.errors:
        print(f"  Errors           : {len(summary.errors)}")
        for e in summary.errors:
            print(f"    - {e}")

    if args.report:
        SessionFactory = init_db(db_path)
        ensure_table(db_path)
        with SessionFactory() as session:
            runs = get_segmentation_runs(session)
        print("\n--- Per-Image Segmentation ---")
        header = f"{'Filename':<25} {'Segmenter':<22} {'Masks':>5}  {'DefectArea%':>12}"
        print(header)
        print("-" * 70)
        for r in runs:
            print(
                f"{r.source_filename:<25} {r.segmenter_name:<22} {r.n_masks:>5}"
                f"  {r.composite_defect_area_pct:>11.2f}%"
            )


if __name__ == "__main__":
    main()
