"""Pixel-level IoU/Dice evaluation of the segmentation stage's COMPOSITE mask
output against hand-drawn polygon ground truth (see data/annotations/README.md).

Only the composite mask is scored — there is no per-instance predicted mask
artifact on disk to match against (SegmentationMask.to_dict() stores only
summary stats), so this mirrors src/detection/evaluation.py's structure but
at the pixel level instead of the box level.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .metrics import union_mask


def polygons_to_mask(
    polygons: list[list[tuple[float, float]]],
    height: int,
    width: int,
) -> np.ndarray:
    """Rasterize a list of point-polygons and union them into one binary mask
    (uint8, 0/255), matching the union semantics of segment_all()'s composite.
    Polygons with fewer than 3 points are ignored as degenerate.
    """
    canvases = []
    for poly in polygons:
        if len(poly) < 3:
            continue
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        canvas = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(canvas, [pts], 255)
        canvases.append(canvas)
    if not canvases:
        return np.zeros((height, width), dtype=np.uint8)
    return union_mask(canvases)


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection-over-union of two binary masks. Both empty -> 1.0 (vacuous agreement)."""
    p, g = pred > 0, gt > 0
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(p, g).sum() / union)


def mask_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice coefficient (pixel-level F1) of two binary masks. Both empty -> 1.0."""
    p, g = pred > 0, gt > 0
    denom = p.sum() + g.sum()
    if denom == 0:
        return 1.0
    return float(2 * np.logical_and(p, g).sum() / denom)


@dataclass
class ImageMaskResult:
    """Outcome of scoring one image's predicted composite mask against ground truth."""
    source_filename: str
    iou: float
    dice: float
    gt_area_px: int
    pred_area_px: int
    intersection_px: int


def score_masks(pred: np.ndarray, gt: np.ndarray, source_filename: str) -> ImageMaskResult:
    p, g = pred > 0, gt > 0
    return ImageMaskResult(
        source_filename=source_filename,
        iou=mask_iou(pred, gt),
        dice=mask_dice(pred, gt),
        gt_area_px=int(g.sum()),
        pred_area_px=int(p.sum()),
        intersection_px=int(np.logical_and(p, g).sum()),
    )


@dataclass
class AggregateMaskMetrics:
    """Precision/recall/dice are pooled (pixel counts summed across images before
    dividing); mean_iou is macro-averaged (mean of per-image IoU) — this mirrors
    the mixed pooled/macro convention already used by
    src/detection/evaluation.py::aggregate_metrics.
    """
    precision: float
    recall: float
    dice: float
    mean_iou: float
    n_images: int


def aggregate_mask_metrics(results: list[ImageMaskResult]) -> AggregateMaskMetrics:
    if not results:
        return AggregateMaskMetrics(0.0, 0.0, 0.0, 0.0, 0)

    total_inter = sum(r.intersection_px for r in results)
    total_pred = sum(r.pred_area_px for r in results)
    total_gt = sum(r.gt_area_px for r in results)

    precision = total_inter / total_pred if total_pred > 0 else 0.0
    recall = total_inter / total_gt if total_gt > 0 else 0.0
    dice = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    mean_iou = sum(r.iou for r in results) / len(results)

    return AggregateMaskMetrics(
        precision=precision, recall=recall, dice=dice,
        mean_iou=mean_iou, n_images=len(results),
    )
