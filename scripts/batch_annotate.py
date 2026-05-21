"""CLI: produce annotated overlay images from the full pipeline DB."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate GP screen analysis results.")
    parser.add_argument("--db", default="data/processed/images.db")
    parser.add_argument("--overlays-dir", default="outputs/overlays")
    parser.add_argument("--panels-dir", default="outputs/panels")
    parser.add_argument("--no-panels", action="store_true", help="Skip 3-panel composites")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    from src.annotation import annotate_all, get_all_annotations, ensure_table
    from src.ingestion.database import init_db

    summary = annotate_all(
        db_path=root / args.db,
        overlay_dir=root / args.overlays_dir,
        panels_dir=root / args.panels_dir if not args.no_panels else None,
        include_panels=not args.no_panels,
    )

    print("\n--- Annotation Summary ---")
    print(f"  Images processed : {summary.processed}/{summary.total}")
    print(f"  Skipped          : {summary.skipped}")
    if summary.errors:
        print(f"  Errors           : {len(summary.errors)}")
        for e in summary.errors:
            print(f"    - {e}")

    if args.report:
        SessionFactory = init_db(root / args.db)
        ensure_table(root / args.db)
        with SessionFactory() as session:
            recs = get_all_annotations(session)
        print("\n--- Annotation Outputs ---")
        for r in recs:
            ann = Path(r.annotated_path).name if r.annotated_path else "—"
            pan = Path(r.panel_path).name if r.panel_path else "—"
            print(f"  {r.source_filename:<25}  overlay={ann}  panel={pan}")


if __name__ == "__main__":
    main()
