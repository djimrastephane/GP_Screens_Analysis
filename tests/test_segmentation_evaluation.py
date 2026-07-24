"""Unit tests for src/segmentation/evaluation.py (pixel IoU/Dice scoring)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.segmentation.evaluation import (
    polygons_to_mask, mask_iou, mask_dice, score_masks, aggregate_mask_metrics,
)


def _square_mask(h: int, w: int, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    m[y1:y2, x1:x2] = 255
    return m


class TestPolygonsToMask:
    def test_fills_square_polygon(self):
        poly = [(10, 10), (30, 10), (30, 30), (10, 30)]
        mask = polygons_to_mask([poly], height=50, width=50)
        assert mask.shape == (50, 50)
        assert (mask > 0).sum() > 0
        # roughly a 20x20 square (fillPoly is corner-inclusive, so slightly larger)
        assert 380 <= (mask > 0).sum() <= 441

    def test_unions_overlapping_polygons(self):
        poly_a = [(0, 0), (20, 0), (20, 20), (0, 20)]
        poly_b = [(10, 10), (30, 10), (30, 30), (10, 30)]
        mask = polygons_to_mask([poly_a, poly_b], height=50, width=50)
        area_a = polygons_to_mask([poly_a], height=50, width=50)
        area_b = polygons_to_mask([poly_b], height=50, width=50)
        union_area = (mask > 0).sum()
        assert union_area > max((area_a > 0).sum(), (area_b > 0).sum())
        assert union_area < (area_a > 0).sum() + (area_b > 0).sum()

    def test_degenerate_polygon_ignored(self):
        mask = polygons_to_mask([[(0, 0), (10, 10)]], height=50, width=50)
        assert (mask > 0).sum() == 0

    def test_empty_polygon_list_returns_empty_mask(self):
        mask = polygons_to_mask([], height=50, width=50)
        assert mask.shape == (50, 50)
        assert (mask > 0).sum() == 0


class TestMaskIoUAndDice:
    def test_perfect_overlap_is_one(self):
        m = _square_mask(50, 50, 10, 10, 30, 30)
        assert mask_iou(m, m) == 1.0
        assert mask_dice(m, m) == 1.0

    def test_disjoint_masks_is_zero(self):
        a = _square_mask(50, 50, 0, 0, 10, 10)
        b = _square_mask(50, 50, 20, 20, 30, 30)
        assert mask_iou(a, b) == 0.0
        assert mask_dice(a, b) == 0.0

    def test_both_empty_is_one(self):
        empty = np.zeros((50, 50), dtype=np.uint8)
        assert mask_iou(empty, empty) == 1.0
        assert mask_dice(empty, empty) == 1.0

    def test_known_partial_overlap(self):
        # a: 0..20 x 0..20 (400 px), b: 10..30 x 0..20 (400 px)
        # intersection: 10..20 x 0..20 = 200 px, union = 400+400-200 = 600 px
        a = _square_mask(50, 50, 0, 0, 20, 20)
        b = _square_mask(50, 50, 10, 0, 30, 20)
        assert mask_iou(a, b) == 200 / 600
        assert mask_dice(a, b) == 2 * 200 / (400 + 400)


class TestScoreMasks:
    def test_returns_correct_areas(self):
        pred = _square_mask(50, 50, 0, 0, 20, 20)   # 400 px
        gt = _square_mask(50, 50, 10, 0, 30, 20)     # 400 px, 200 px overlap
        result = score_masks(pred, gt, "test.jpg")
        assert result.source_filename == "test.jpg"
        assert result.pred_area_px == 400
        assert result.gt_area_px == 400
        assert result.intersection_px == 200
        assert result.iou == 200 / 600


class TestAggregateMaskMetrics:
    def test_empty_results_give_zero_metrics_not_a_crash(self):
        metrics = aggregate_mask_metrics([])
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.dice == 0.0
        assert metrics.mean_iou == 0.0
        assert metrics.n_images == 0

    def test_perfect_predictions_give_one(self):
        m = _square_mask(50, 50, 10, 10, 30, 30)
        result = score_masks(m, m, "a.jpg")
        metrics = aggregate_mask_metrics([result])
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.dice == 1.0
        assert metrics.mean_iou == 1.0
        assert metrics.n_images == 1

    def test_pooled_precision_recall_across_two_images(self):
        # Image 1: pred=100px, gt=100px, intersection=100px (perfect)
        pred1 = _square_mask(50, 50, 0, 0, 10, 10)
        gt1 = _square_mask(50, 50, 0, 0, 10, 10)
        r1 = score_masks(pred1, gt1, "a.jpg")

        # Image 2: pred=400px, gt=100px, intersection=100px (over-segmented)
        pred2 = _square_mask(50, 50, 0, 0, 20, 20)
        gt2 = _square_mask(50, 50, 0, 0, 10, 10)
        r2 = score_masks(pred2, gt2, "b.jpg")

        metrics = aggregate_mask_metrics([r1, r2])
        # pooled: total_inter=200, total_pred=500, total_gt=200
        assert metrics.precision == 200 / 500
        assert metrics.recall == 200 / 200
        assert metrics.n_images == 2
