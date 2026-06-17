"""IoU calculation, containment ratio, and multi-stage non-max suppression."""

from __future__ import annotations

from .base import Detection


def iou(a: Detection, b: Detection) -> float:
    """Intersection over union for two detections."""
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h

    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def containment_ratio(inner: Detection, outer: Detection) -> float:
    """Fraction of inner's box area that lies within outer's box.

    Returns 1.0 if inner is completely inside outer, 0.0 if no overlap.
    """
    ix1 = max(inner.x1, outer.x1)
    iy1 = max(inner.y1, outer.y1)
    ix2 = min(inner.x2, outer.x2)
    iy2 = min(inner.y2, outer.y2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h

    inner_area = (inner.x2 - inner.x1) * (inner.y2 - inner.y1)
    return inter / inner_area if inner_area > 0 else 0.0


def nms(
    detections: list[Detection],
    iou_threshold: float = 0.5,
) -> list[Detection]:
    """Greedy per-class NMS: keep highest-confidence box, suppress overlapping lower ones."""
    if not detections:
        return []

    classes = {d.class_id for d in detections}
    kept: list[Detection] = []

    for cls in classes:
        cls_dets = sorted(
            [d for d in detections if d.class_id == cls],
            key=lambda d: d.confidence,
            reverse=True,
        )
        while cls_dets:
            best = cls_dets.pop(0)
            kept.append(best)
            cls_dets = [d for d in cls_dets if iou(best, d) < iou_threshold]

    return kept


def nms_global(
    detections: list[Detection],
    iou_threshold: float = 0.35,
    cross_iou_threshold: float = 0.50,
    containment_threshold: float = 0.80,
) -> list[Detection]:
    """Three-stage suppression that eliminates cross-class overlap clutter.

    Stage 1 — Per-class greedy NMS (standard within-class suppression).
    Stage 2 — Cross-class containment: suppress any detection whose bounding
               box is more than containment_threshold contained inside a larger
               detection from any other class. Keeps the larger detection.
    Stage 3 — Cross-class IoU: for any pair of detections from different classes
               whose IoU exceeds cross_iou_threshold, suppress the lower-confidence
               one. Eliminates duplicate boxes on the same damage region.
    """
    if not detections:
        return []

    # Stage 1: per-class NMS
    after_cls_nms = nms(detections, iou_threshold)

    # Stage 2: cross-class containment suppression
    # Sort by area descending so larger detections protect smaller ones nested inside
    # them. Ties (e.g. two detectors firing on the identical box) break on confidence,
    # so the more confident detection survives instead of an arbitrary class-id order.
    sorted_by_area = sorted(
        after_cls_nms,
        key=lambda d: ((d.x2 - d.x1) * (d.y2 - d.y1), d.confidence),
        reverse=True,
    )
    suppressed: set[int] = set()
    for i, outer in enumerate(sorted_by_area):
        if i in suppressed:
            continue
        for j, inner in enumerate(sorted_by_area):
            if j <= i or j in suppressed:
                continue
            if inner.class_id == outer.class_id:
                continue  # already handled by per-class NMS
            if containment_ratio(inner, outer) >= containment_threshold:
                # inner is mostly inside outer — suppress inner
                suppressed.add(j)

    after_containment = [d for i, d in enumerate(sorted_by_area) if i not in suppressed]

    # Stage 3: cross-class IoU suppression
    after_containment_sorted = sorted(
        after_containment, key=lambda d: d.confidence, reverse=True
    )
    final_suppressed: set[int] = set()
    for i, a in enumerate(after_containment_sorted):
        if i in final_suppressed:
            continue
        for j, b in enumerate(after_containment_sorted):
            if j <= i or j in final_suppressed:
                continue
            if a.class_id == b.class_id:
                continue  # already handled
            if iou(a, b) >= cross_iou_threshold:
                # Keep a (higher confidence); suppress b
                final_suppressed.add(j)

    return [d for i, d in enumerate(after_containment_sorted) if i not in final_suppressed]
