"""CLI: preprocess all ingested images and persist results to SQLite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import get_all_preprocessed, ensure_table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess ingested GP screen images."
    )
    parser.add_argument(
        "--db", default="data/processed/images.db", help="SQLite database path"
    )
    parser.add_argument(
        "--output", default="data/processed", help="Output directory root"
    )
    parser.add_argument(
        "--config",
        default="configs/preprocessing_config.yaml",
        help="Preprocessing config YAML",
    )
    parser.add_argument(
        "--report", action="store_true", help="Print all preprocessed records after run"
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    db_path = project_root / args.db
    output_dir = project_root / args.output
    config_path = project_root / args.config

    from src.preprocessing.pipeline import preprocess_all

    summary = preprocess_all(db_path, output_dir, config_path)

    print("\n--- Preprocessing Summary ---")
    print(f"  Total images      : {summary.total}")
    print(f"  Processed         : {summary.processed}")
    print(f"  Enhanced          : {summary.enhanced}")
    print(f"  Skipped           : {summary.skipped_not_processable}")
    if summary.errors:
        print(f"  Errors            : {len(summary.errors)}")
        for e in summary.errors:
            print(f"    - {e}")

    if args.report:
        SessionFactory = ensure_table(db_path)
        with SessionFactory() as session:
            records = get_all_preprocessed(session)
        print("\n--- Preprocessed Records ---")
        header = f"{'Filename':<25} {'Original':>12} {'Target':>10} {'Scale':>7}  Enhancements"
        print(header)
        print("-" * len(header))
        for r in records:
            orig = f"{r.original_width}x{r.original_height}"
            tgt = f"{r.target_width}x{r.target_height}"
            print(
                f"{r.source_filename:<25} {orig:>12} {tgt:>10} {r.scale:>7.4f}  {r.enhancements_applied}"
            )


if __name__ == "__main__":
    main()
