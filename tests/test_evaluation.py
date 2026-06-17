"""Unit tests for src/detection/evaluation.py (precision/recall/F1/IoU scoring)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.evaluation import match_detections, aggregate_metrics


def _box(x1, y1, x2, y2, confidence=1.0):
    return SimpleNamespace(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence)


class TestMatchDetections:
    def test_perfect_match_is_one_tp(self):
        pred = [_box(0, 0, 100, 100)]
        gt = [_box(0, 0, 100, 100)]
        result = match_detections(pred, gt, iou_threshold=0.5)
        assert result.true_positives == 1
        assert result.false_positives == 0
        assert result.false_negatives == 0
        assert result.matched_ious == [1.0]

    def test_no_predictions_is_all_false_negatives(self):
        gt = [_box(0, 0, 100, 100), _box(200, 200, 300, 300)]
        result = match_detections([], gt, iou_threshold=0.5)
        assert result.true_positives == 0
        assert result.false_negatives == 2

    def test_no_ground_truth_is_all_false_positives(self):
        pred = [_box(0, 0, 100, 100)]
        result = match_detections(pred, [], iou_threshold=0.5)
        assert result.true_positives == 0
        assert result.false_positives == 1

    def test_low_overlap_below_threshold_counts_as_fp_and_fn(self):
        pred = [_box(0, 0, 50, 50)]
        gt = [_box(40, 40, 140, 140)]  # small overlap, IoU well under 0.5
        result = match_detections(pred, gt, iou_threshold=0.5)
        assert result.true_positives == 0
        assert result.false_positives == 1
        assert result.false_negatives == 1

    def test_each_prediction_claims_a_distinct_gt_box(self):
        pred = [_box(0, 0, 100, 100), _box(200, 200, 300, 300)]
        gt = [_box(0, 0, 100, 100), _box(200, 200, 300, 300)]
        result = match_detections(pred, gt, iou_threshold=0.5)
        assert result.true_positives == 2
        assert result.false_positives == 0
        assert result.false_negatives == 0

    def test_extra_prediction_on_same_box_is_a_false_positive(self):
        pred = [_box(0, 0, 100, 100), _box(1, 1, 100, 100)]
        gt = [_box(0, 0, 100, 100)]
        result = match_detections(pred, gt, iou_threshold=0.5)
        assert result.true_positives == 1
        assert result.false_positives == 1


class TestAggregateMetrics:
    def test_perfect_predictions_give_precision_recall_f1_of_one(self):
        result = match_detections([_box(0, 0, 100, 100)], [_box(0, 0, 100, 100)])
        metrics = aggregate_metrics([result])
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1 == 1.0
        assert metrics.mean_iou == 1.0

    def test_empty_results_give_zero_metrics_not_a_crash(self):
        metrics = aggregate_metrics([])
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1 == 0.0
        assert metrics.mean_iou == 0.0

    def test_sums_across_multiple_images(self):
        perfect = match_detections([_box(0, 0, 100, 100)], [_box(0, 0, 100, 100)])
        missed = match_detections([], [_box(0, 0, 100, 100)])
        metrics = aggregate_metrics([perfect, missed])
        assert metrics.true_positives == 1
        assert metrics.false_negatives == 1
        assert metrics.recall == 0.5
