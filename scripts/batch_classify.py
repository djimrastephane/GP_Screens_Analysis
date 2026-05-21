"""CLI: classify detected GP screen failure regions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify GP screen failure types.")
    parser.add_argument("--db", default="data/processed/images.db")
    parser.add_argument("--config", default="configs/severity_config.yaml")
    parser.add_argument("--detector", default="ContourDetector")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    db_path = root / args.db
    config_path = root / args.config

    from src.classification import classify_all, get_classification_runs, ensure_table
    from src.ingestion.database import init_db

    summary = classify_all(db_path, config_path, args.detector)

    print("\n--- Classification Summary ---")
    print(f"  Images processed    : {summary.processed}/{summary.total}")
    print(f"  Total classifications: {summary.total_classifications}")
    print(f"  Require review      : {summary.requires_review}")
    print(f"  Failure type counts :")
    for ft, count in sorted(summary.failure_type_counts.items(), key=lambda x: -x[1]):
        print(f"    {ft:<22} {count:>4}")
    if summary.errors:
        print(f"  Errors: {len(summary.errors)}")

    if args.report:
        SessionFactory = init_db(db_path)
        ensure_table(db_path)
        with SessionFactory() as session:
            runs = get_classification_runs(session)
        print("\n--- Per-Image Classification ---")
        header = f"{'Filename':<25} {'Dominant Failure':<22} {'Severity':<10} {'Review':>6}"
        print(header)
        print("-" * 70)
        for r in runs:
            print(
                f"{r.source_filename:<25} {r.dominant_failure_type:<22} "
                f"{r.overall_severity:<10} {r.n_requires_review:>6}"
            )


if __name__ == "__main__":
    main()
