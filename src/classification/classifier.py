"""RuleBasedClassifier: wires features → rules → severity → ClassificationResult."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import BaseClassifier, ClassificationResult, ImageClassificationResult, FailureType
from .features import extract_features
from .rules import classify_features
from .severity import area_to_severity, load_thresholds, worst_severity


class RuleBasedClassifier(BaseClassifier):
    """Deterministic rule-based classifier using shape and colour features."""

    def __init__(self, config_path: str | Path | None = None):
        cfg = Path(config_path) if config_path else None
        self._thresholds = load_thresholds(cfg)
        self._review_threshold = 0.65
        if cfg and cfg.exists():
            import yaml
            with cfg.open() as f:
                raw = yaml.safe_load(f)
            self._review_threshold = raw.get("confidence_review_threshold", 0.65)

    @property
    def name(self) -> str:
        return "RuleBasedClassifier"

    def classify_one(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        detection: dict,
        composite_defect_area_pct: float,
    ) -> ClassificationResult:
        fv = extract_features(img, mask, detection)
        severity = area_to_severity(
            fv.area_pct,
            failure_type="",
            thresholds=self._thresholds,
        )
        result = classify_features(
            fv,
            detection_index=detection.get("detection_index", 0),
            class_name=detection.get("class_name", "unknown"),
            severity=severity,
            review_threshold=self._review_threshold,
        )
        # Re-apply severity now that failure_type is known (escalation for collapse/plugging)
        result.severity = area_to_severity(
            fv.area_pct,
            failure_type=result.failure_type.value,
            thresholds=self._thresholds,
        )
        return result

    def classify_image(
        self,
        image_id: str,
        source_filename: str,
        img: np.ndarray,
        detections: list[dict],
        masks: list[np.ndarray],
        composite_defect_area_pct: float,
        run_timestamp: str,
    ) -> ImageClassificationResult:
        """Classify all detections for one image."""
        classifications: list[ClassificationResult] = []

        for idx, (det, mask) in enumerate(zip(detections, masks)):
            det_with_idx = {**det, "detection_index": idx}
            result = self.classify_one(img, mask, det_with_idx, composite_defect_area_pct)
            classifications.append(result)

        # Dominant failure type = most frequent non-unknown
        type_counts: dict[str, int] = {}
        for c in classifications:
            t = c.failure_type.value
            if t != FailureType.UNKNOWN.value:
                type_counts[t] = type_counts.get(t, 0) + 1
        dominant = (
            max(type_counts, key=type_counts.__getitem__)
            if type_counts else FailureType.UNKNOWN.value
        )

        overall_sev = worst_severity([c.severity for c in classifications])

        return ImageClassificationResult(
            image_id=image_id,
            source_filename=source_filename,
            classifier_name=self.name,
            classifications=classifications,
            run_timestamp=run_timestamp,
            dominant_failure_type=dominant,
            composite_defect_area_pct=composite_defect_area_pct,
            overall_severity=overall_sev,
        )
