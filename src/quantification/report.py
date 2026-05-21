"""QuantificationReport dataclass and per-defect measurements."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .measurements import equivalent_diameter_px, pixels_to_mm
from .aggregator import AggregatedMetrics, FailureTypeStats


@dataclass
class DefectMeasurement:
    """Measurements for one detected defect."""
    detection_index: int
    failure_type: str
    severity: str
    confidence: float
    requires_human_review: bool
    # Pixel measurements
    defect_area_px: int
    defect_area_pct_of_screen: float
    equivalent_diameter_px: float
    fill_ratio: float
    # Physical measurements (None when not calibrated)
    equivalent_diameter_mm: float | None

    def to_dict(self) -> dict:
        return {
            "detection_index": self.detection_index,
            "failure_type": self.failure_type,
            "severity": self.severity,
            "confidence": round(self.confidence, 4),
            "requires_human_review": self.requires_human_review,
            "defect_area_px": self.defect_area_px,
            "defect_area_pct_of_screen": round(self.defect_area_pct_of_screen, 4),
            "equivalent_diameter_px": round(self.equivalent_diameter_px, 2),
            "fill_ratio": round(self.fill_ratio, 4),
            "equivalent_diameter_mm": (
                round(self.equivalent_diameter_mm, 2)
                if self.equivalent_diameter_mm is not None else None
            ),
        }


@dataclass
class QuantificationReport:
    """Complete quantification output for one image."""
    image_id: str
    source_filename: str
    image_width: int
    image_height: int
    screen_region_area_px: int
    scale_calibrated: bool
    pixels_per_mm: float | None
    # Aggregate metrics
    n_defects: int
    composite_defect_area_px: int
    erosion_pct: float              # relative to screen content area
    defect_density_per_10k_px: float
    n_requires_review: int
    overall_severity: str
    dominant_failure_type: str
    # Breakdowns
    failure_type_breakdown: dict[str, dict]   # {ft: {count, area_px, area_pct_of_screen}}
    severity_breakdown: dict[str, int]
    # Per-defect detail
    defects: list[DefectMeasurement]
    run_timestamp: str

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "source_filename": self.source_filename,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "screen_region_area_px": self.screen_region_area_px,
            "scale_calibrated": self.scale_calibrated,
            "pixels_per_mm": self.pixels_per_mm,
            "n_defects": self.n_defects,
            "composite_defect_area_px": self.composite_defect_area_px,
            "erosion_pct": round(self.erosion_pct, 4),
            "defect_density_per_10k_px": round(self.defect_density_per_10k_px, 4),
            "n_requires_review": self.n_requires_review,
            "overall_severity": self.overall_severity,
            "dominant_failure_type": self.dominant_failure_type,
            "failure_type_breakdown": self.failure_type_breakdown,
            "severity_breakdown": self.severity_breakdown,
            "defects": [d.to_dict() for d in self.defects],
            "run_timestamp": self.run_timestamp,
        }

    def to_csv_row(self) -> dict:
        """Flat row for summary CSV export."""
        return {
            "image_id": self.image_id,
            "source_filename": self.source_filename,
            "n_defects": self.n_defects,
            "erosion_pct": round(self.erosion_pct, 2),
            "composite_defect_area_px": self.composite_defect_area_px,
            "screen_region_area_px": self.screen_region_area_px,
            "defect_density_per_10k_px": round(self.defect_density_per_10k_px, 4),
            "dominant_failure_type": self.dominant_failure_type,
            "overall_severity": self.overall_severity,
            "n_requires_review": self.n_requires_review,
            "scale_calibrated": self.scale_calibrated,
        }


def build_report(
    image_id: str,
    source_filename: str,
    image_width: int,
    image_height: int,
    screen_area_px: int,
    classifications: list[dict],
    seg_masks: list[dict],
    metrics: AggregatedMetrics,
    pixels_per_mm: float | None,
    run_timestamp: str,
) -> QuantificationReport:
    """Construct a QuantificationReport from aggregated pipeline data."""
    defects: list[DefectMeasurement] = []

    for cls, seg in zip(classifications, seg_masks):
        area_px = seg.get("defect_area_px", 0)
        diam_px = equivalent_diameter_px(area_px)
        diam_mm = (
            pixels_to_mm(diam_px, pixels_per_mm)
            if pixels_per_mm is not None else None
        )
        area_pct = area_px / screen_area_px * 100.0 if screen_area_px > 0 else 0.0

        defects.append(DefectMeasurement(
            detection_index=cls.get("detection_index", 0),
            failure_type=cls.get("failure_type", "unknown"),
            severity=cls.get("severity", "low"),
            confidence=cls.get("confidence", 0.0),
            requires_human_review=cls.get("requires_human_review", True),
            defect_area_px=area_px,
            defect_area_pct_of_screen=round(area_pct, 4),
            equivalent_diameter_px=round(diam_px, 2),
            fill_ratio=seg.get("fill_ratio", 0.0),
            equivalent_diameter_mm=round(diam_mm, 2) if diam_mm is not None else None,
        ))

    ft_breakdown = {
        ft: {
            "count": s.count,
            "area_px": s.area_px,
            "area_pct_of_screen": round(s.area_pct_of_screen, 4),
        }
        for ft, s in metrics.failure_type_stats.items()
    }

    return QuantificationReport(
        image_id=image_id,
        source_filename=source_filename,
        image_width=image_width,
        image_height=image_height,
        screen_region_area_px=screen_area_px,
        scale_calibrated=pixels_per_mm is not None,
        pixels_per_mm=pixels_per_mm,
        n_defects=metrics.n_defects,
        composite_defect_area_px=metrics.composite_defect_area_px,
        erosion_pct=metrics.erosion_pct,
        defect_density_per_10k_px=metrics.defect_density_per_10k_px,
        n_requires_review=metrics.n_requires_review,
        overall_severity=metrics.overall_severity,
        dominant_failure_type=metrics.dominant_failure_type,
        failure_type_breakdown=ft_breakdown,
        severity_breakdown=metrics.severity_counts,
        defects=defects,
        run_timestamp=run_timestamp,
    )


def export_csv(reports: list[QuantificationReport], output_path: Path) -> None:
    """Write a flat summary CSV of all quantification reports."""
    if not reports:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r.to_csv_row() for r in reports]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
