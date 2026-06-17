"""Precision / recall / F1 / IoU evaluation of detector output against ground truth.

Ground-truth boxes only need x1/y1/x2/y2 (see data/annotations/README.md for the
on-disk JSON format) — detection is evaluated as class-agnostic localisation,
since failure-type labelling happens downstream in src/classification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .nms import iou


class BoxLike(Protocol):
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class ImageMatchResult:
    """Outcome of matching one image's predicted boxes against its ground truth."""
    true_positives: int
    false_positives: int
    false_negatives: int
    matched_ious: list[float] = field(default_factory=list)


def match_detections(
    predictions: list[BoxLike],
    ground_truth: list[BoxLike],
    iou_threshold: float = 0.5,
) -> ImageMatchResult:
    """Greedy box matching: each prediction claims its best unmatched ground-truth box.

    Predictions are consumed in the order given — callers should sort by
    confidence (highest first) beforehand if predictions carry one.
    """
    unmatched_gt = list(range(len(ground_truth)))
    true_positives = 0
    matched_ious: list[float] = []

    for pred in predictions:
        best_iou = 0.0
        best_idx = -1
        for gt_idx in unmatched_gt:
            score = iou(pred, ground_truth[gt_idx])
            if score > best_iou:
                best_iou = score
                best_idx = gt_idx

        if best_idx != -1 and best_iou >= iou_threshold:
            true_positives += 1
            matched_ious.append(best_iou)
            unmatched_gt.remove(best_idx)

    false_positives = len(predictions) - true_positives
    false_negatives = len(unmatched_gt)

    return ImageMatchResult(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        matched_ious=matched_ious,
    )


@dataclass
class AggregateMetrics:
    precision: float
    recall: float
    f1: float
    mean_iou: float
    true_positives: int
    false_positives: int
    false_negatives: int


def aggregate_metrics(results: list[ImageMatchResult]) -> AggregateMetrics:
    """Sum per-image match results into overall precision/recall/F1/mean IoU."""
    tp = sum(r.true_positives for r in results)
    fp = sum(r.false_positives for r in results)
    fn = sum(r.false_negatives for r in results)
    all_ious = [iou_val for r in results for iou_val in r.matched_ious]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    mean_iou = sum(all_ious) / len(all_ious) if all_ious else 0.0

    return AggregateMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        mean_iou=mean_iou,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
    )
