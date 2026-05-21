"""CLI: generate per-image and summary PDF reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GP screen analysis PDF reports.")
    parser.add_argument("--db", default="data/processed/images.db")
    parser.add_argument("--reports-dir", default="outputs/reports")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    from src.reporting import generate_all_reports, get_reports, ensure_table
    from src.ingestion.database import init_db

    summary = generate_all_reports(root / args.db, root / args.reports_dir)

    print("\n--- Reporting Summary ---")
    print(f"  Per-image PDFs   : {summary.per_image_generated}/{summary.total_images}")
    print(f"  Summary PDF      : {'OK' if summary.summary_generated else 'FAILED'}")
    print(f"  Reports dir      : {summary.report_dir}")
    if summary.errors:
        print(f"  Errors           : {len(summary.errors)}")
        for e in summary.errors:
            print(f"    - {e}")

    if args.report:
        SessionFactory = init_db(root / args.db)
        ensure_table(root / args.db)
        with SessionFactory() as session:
            recs = get_reports(session)
        print("\n--- Generated Reports ---")
        for r in recs:
            size = Path(r.pdf_path).stat().st_size // 1024 if Path(r.pdf_path).exists() else 0
            print(f"  [{r.report_type:<10}] {Path(r.pdf_path).name:<50} {size:>6} KB")


if __name__ == "__main__":
    main()
