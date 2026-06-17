from .base import BaseDetector, Detection, DetectionResult
from .nms import iou, nms
from .contour import ContourDetector, ContourConfig
from .yolo import YOLOv8Detector
from .pipeline import detect_all, DetectionSummary
from .store import ensure_table, get_detection_runs, upsert_detection_run
from .evaluation import match_detections, aggregate_metrics, ImageMatchResult, AggregateMetrics

__all__ = [
    "BaseDetector", "Detection", "DetectionResult",
    "iou", "nms",
    "ContourDetector", "ContourConfig",
    "YOLOv8Detector",
    "detect_all", "DetectionSummary",
    "ensure_table", "get_detection_runs", "upsert_detection_run",
    "match_detections", "aggregate_metrics", "ImageMatchResult", "AggregateMetrics",
]
