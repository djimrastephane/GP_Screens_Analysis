"""Aggregate per-detection data into per-image quantification metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

from .measurements import corrected_erosion_pct, defect_density, equivalent_diameter_px


@dataclass
class FailureTypeStats:
    count: int = 0
    area_px: int = 0
    area_pct_of_screen: float = 0.0
    max_area_px: int = 0


@dataclass
class AggregatedMetrics:
    """Summary metrics derived from all detections in one image."""
    n_defects: int
    composite_defect_area_px: int
    composite_defect_area_pct: float    # relative to screen content area
    erosion_pct: float                  # same as composite_defect_area_pct (explicit name)
    defect_density_per_10k_px: float
    n_requires_review: int
    largest_defect_area_px: int
    largest_defect_failure_type: str
    overall_severity: str
    dominant_failure_type: str
    failure_type_stats: dict[str, FailureTypeStats]
    severity_counts: dict[str, int]


def aggregate(
    classifications: list[dict],
    seg_masks: list[dict],          # per_mask_json entries from segmentation
    composite_defect_area_px: int,
    screen_area_px: int,
    overall_severity: str,
    dominant_failure_type: str,
) -> AggregatedMetrics:
    """Combine classification and segmentation data into aggregate metrics.

    Args:
        classifications: list of ClassificationResult.to_dict() entries.
        seg_masks: list of SegmentationMask.to_dict() entries (parallel with classifications).
        composite_defect_area_px: union of all mask pixel areas.
        screen_area_px: content area excluding letterbox padding.
        overall_severity: worst severity from classification.
        dominant_failure_type: most frequent failure type from classification.
    """
    n_defects = len(classifications)
    erosion = corrected_erosion_pct(composite_defect_area_px, screen_area_px)
    density = defect_density(n_defects, screen_area_px)
    n_review = sum(1 for c in classifications if c.get("requires_human_review"))

    failure_stats: dict[str, FailureTypeStats] = {}
    severity_counts: dict[str, int] = {}
    largest_area = 0
    largest_type = "unknown"

    for cls, seg in zip(classifications, seg_masks):
        ft = cls.get("failure_type", "unknown")
        sev = cls.get("severity", "low")
        area = seg.get("defect_area_px", 0)
        area_pct = corrected_erosion_pct(area, screen_area_px)

        if ft not in failure_stats:
            failure_stats[ft] = FailureTypeStats()
        failure_stats[ft].count += 1
        failure_stats[ft].area_px += area
        failure_stats[ft].area_pct_of_screen += area_pct
        failure_stats[ft].max_area_px = max(failure_stats[ft].max_area_px, area)

        severity_counts[sev] = severity_counts.get(sev, 0) + 1

        if area > largest_area:
            largest_area = area
            largest_type = ft

    return AggregatedMetrics(
        n_defects=n_defects,
        composite_defect_area_px=composite_defect_area_px,
        composite_defect_area_pct=round(erosion, 4),
        erosion_pct=round(erosion, 4),
        defect_density_per_10k_px=round(density, 4),
        n_requires_review=n_review,
        largest_defect_area_px=largest_area,
        largest_defect_failure_type=largest_type,
        overall_severity=overall_severity,
        dominant_failure_type=dominant_failure_type,
        failure_type_stats=failure_stats,
        severity_counts=severity_counts,
    )
