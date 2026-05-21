"""CLI: compute quantification reports from the full pipeline DB."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantify GP screen defect measurements.")
    parser.add_argument("--db", default="data/processed/images.db")
    parser.add_argument("--config", default="configs/severity_config.yaml")
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    db_path = root / args.db
    config_path = root / args.config
    output_dir = root / args.output

    from src.quantification import quantify_all, get_all_reports, ensure_table
    from src.ingestion.database import init_db

    summary = quantify_all(db_path, output_dir, config_path)

    print("\n--- Quantification Summary ---")
    print(f"  Images processed : {summary.processed}/{summary.total}")
    print(f"  Skipped          : {summary.skipped}")
    if summary.csv_path:
        print(f"  CSV exported     : {summary.csv_path}")
    if summary.errors:
        print(f"  Errors           : {len(summary.errors)}")

    if args.report:
        SessionFactory = init_db(db_path)
        ensure_table(db_path)
        with SessionFactory() as session:
            records = get_all_reports(session)
        print("\n--- Per-Image Quantification ---")
        header = (
            f"{'Filename':<25} {'Erosion%':>9} {'Defects':>7} "
            f"{'ScreenPx':>10} {'Density':>8}  {'Severity':<10} {'Review':>6}"
        )
        print(header)
        print("-" * 85)
        for r in records:
            print(
                f"{r.source_filename:<25} {r.erosion_pct:>8.2f}% {r.n_defects:>7}"
                f" {r.screen_region_area_px:>10} {r.defect_density_per_10k_px:>8.4f}"
                f"  {r.overall_severity:<10} {r.n_requires_review:>6}"
            )


if __name__ == "__main__":
    main()
