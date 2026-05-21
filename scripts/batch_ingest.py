"""CLI: ingest all images from a directory and persist metadata to SQLite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion import get_all_images, ingest_directory, init_db


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest GP screen images into SQLite.")
    parser.add_argument(
        "--input", default="Image", help="Source image directory (default: Image/)"
    )
    parser.add_argument(
        "--db", default="data/processed/images.db", help="SQLite database path"
    )
    parser.add_argument(
        "--config", default="configs/severity_config.yaml", help="Config YAML path"
    )
    parser.add_argument(
        "--report", action="store_true", help="Print all ingested records after run"
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    image_dir = project_root / args.input
    db_path = project_root / args.db
    config_path = project_root / args.config

    db_path.parent.mkdir(parents=True, exist_ok=True)

    summary = ingest_directory(image_dir, db_path, config_path)

    print("\n--- Ingestion Summary ---")
    print(f"  Discovered   : {summary.total_discovered}")
    print(f"  Ingested     : {summary.ingested}")
    print(f"  Skipped      : {summary.skipped_invalid}")
    print(f"  Flagged      : {summary.flagged}")
    print(f"  Not processable: {summary.not_processable}")
    if summary.errors:
        print(f"  Errors       : {len(summary.errors)}")
        for e in summary.errors:
            print(f"    - {e}")

    if args.report:
        SessionFactory = init_db(db_path)
        with SessionFactory() as session:
            records = get_all_images(session)
        print("\n--- Image Records ---")
        header = f"{'Filename':<25} {'Size':>8} {'Resolution':>12} {'Quality Flag'}"
        print(header)
        print("-" * len(header))
        for r in records:
            print(
                f"{r.source_filename:<25} {r.file_size_kb:>6.1f}KB "
                f"{r.width}x{r.height:>4}  {r.quality_flag}"
            )


if __name__ == "__main__":
    main()
