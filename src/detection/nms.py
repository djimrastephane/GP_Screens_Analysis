"""IoU calculation and non-max suppression."""

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


def nms(
    detections: list[Detection],
    iou_threshold: float = 0.5,
) -> list[Detection]:
    """Greedy NMS: keep highest-confidence box, suppress overlapping lower-confidence ones.

    Operates per class_id so detections of different classes are not suppressed
    against each other.
    """
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
