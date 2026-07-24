"""Rule-based failure type classifier.

Rules are applied in priority order. Colour evidence (hue/saturation) takes
precedence over shape because the data shows that corrosion manifests as
elongated dark bands with orange pixels — these would be misclassified as
erosion holes or plugging if shape rules ran first.

Rule order:
  1. Colour override  — orange/rust hue + saturation → corrosion_pitting
  2. Colour anomaly source → corrosion_pitting (explicit colour detection)
  3. Screen collapse  — large area + low solidity
  4. Plugging complete — very large area, dense fill
  5. Plugging partial  — very wide elongated band
  6. Wire wrap failure — elongated + low circularity + not orange
  7. Erosion hole      — compact circular dark blob (default dark_blob)
  8. Mechanical damage — fallback for anything with unusual brightness/hue
  9. Unknown           — no rule matched with adequate confidence

Each rule's confidence() is a hand-weighted blend of the shape/colour
evidence that made it match (hue distance, saturation, area, solidity,
aspect ratio, circularity) — a heuristic ranking signal for triage and the
review-threshold gate, not a calibrated probability. Calibrating against
labelled outcomes requires the ground-truth data tracked in
data/annotations/, which doesn't exist yet.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import ClassificationResult, FailureType, FeatureVector

# Confidence floor: classifications below this get requires_human_review=True
_REVIEW_THRESHOLD = 0.65

# Orange/rust hue range in OpenCV HSV (H is 0–180)
_RUST_HUE_MIN = 5
_RUST_HUE_MAX = 35
_RUST_SAT_MIN = 60


@dataclass
class _Rule:
    failure_type: FailureType
    reasoning: str

    def confidence(self, fv: FeatureVector) -> float:
        raise NotImplementedError


class _CorrosionByColour(_Rule):
    """Orange hue + moderate saturation — overrides detection source."""

    def matches(self, fv: FeatureVector) -> bool:
        return (
            _RUST_HUE_MIN <= fv.mean_hue <= _RUST_HUE_MAX
            and fv.mean_saturation >= _RUST_SAT_MIN
        )

    def confidence(self, fv: FeatureVector) -> float:
        hue_score = 1.0 - abs(fv.mean_hue - 20) / 20.0   # peaks at hue=20°
        sat_score = min(1.0, fv.mean_saturation / 200.0)
        return round(min(1.0, 0.35 + 0.4 * hue_score + 0.25 * sat_score), 4)


class _CorrosionBySource(_Rule):
    """Explicit colour-anomaly detection with any hue."""

    def matches(self, fv: FeatureVector) -> bool:
        return fv.detection_source == "contour_color"

    def confidence(self, fv: FeatureVector) -> float:
        return round(min(1.0, 0.45 + 0.55 * fv.mean_saturation / 255.0), 4)


class _ScreenCollapse(_Rule):
    """Large irregular dark region — structural failure."""

    def matches(self, fv: FeatureVector) -> bool:
        return fv.area_pct > 4.0 and fv.solidity < 0.55

    def confidence(self, fv: FeatureVector) -> float:
        area_score = min(1.0, fv.area_pct / 10.0)
        irr_score = 1.0 - fv.solidity
        return round(min(1.0, 0.40 + 0.35 * area_score + 0.25 * irr_score), 4)


class _PluggingComplete(_Rule):
    """Very large dense dark area — complete blockage."""

    def matches(self, fv: FeatureVector) -> bool:
        return fv.area_pct > 8.0 and fv.extent > 0.55

    def confidence(self, fv: FeatureVector) -> float:
        return round(min(1.0, 0.45 + 0.55 * min(1.0, fv.area_pct / 15.0)), 4)


class _PluggingPartial(_Rule):
    """Wide elongated horizontal band — partial blockage across the screen."""

    def matches(self, fv: FeatureVector) -> bool:
        return fv.aspect_ratio > 3.5

    def confidence(self, fv: FeatureVector) -> float:
        ar_score = min(1.0, (fv.aspect_ratio - 3.5) / 3.0)
        return round(min(1.0, 0.42 + 0.58 * ar_score), 4)


class _WireWrapFailure(_Rule):
    """Elongated low-circularity region — wire wrap damage."""

    def matches(self, fv: FeatureVector) -> bool:
        return (
            fv.aspect_ratio > 2.0
            and fv.circularity < 0.40
            and not (_RUST_HUE_MIN <= fv.mean_hue <= _RUST_HUE_MAX
                     and fv.mean_saturation >= _RUST_SAT_MIN)
        )

    def confidence(self, fv: FeatureVector) -> float:
        return round(min(1.0, 0.40 + 0.35 * (1 - fv.circularity) + 0.25 * min(1.0, fv.aspect_ratio / 4.0)), 4)


class _ErosionHole(_Rule):
    """Compact, roughly circular dark blob — the default for dark_blob."""

    def matches(self, fv: FeatureVector) -> bool:
        return (
            fv.detection_source == "contour_dark"
            and fv.circularity >= 0.25
            and fv.aspect_ratio < 3.0
        )

    def confidence(self, fv: FeatureVector) -> float:
        circ_score = min(1.0, fv.circularity / 0.6)
        dark_score = max(0.0, 1.0 - fv.mean_value / 180.0)
        return round(min(1.0, 0.30 + 0.45 * circ_score + 0.25 * dark_score), 4)


class _MechanicalDamage(_Rule):
    """Fallback for blobs with unusual brightness or shape."""

    def matches(self, fv: FeatureVector) -> bool:
        return True  # always matches — used as penultimate fallback

    def confidence(self, fv: FeatureVector) -> float:
        # No specific pattern matched, so this label is inherently uncertain
        # — capped well below the other rules. But it should still track
        # evidence rather than being a flat guess: a solid, well-filled blob
        # is more likely a real (if unclassified) defect than noise that
        # happened to clear the earlier size/aspect filters.
        shape_score = 0.5 * fv.extent + 0.5 * fv.solidity
        return round(min(0.60, 0.20 + 0.40 * shape_score), 4)


# Ordered rule list — first match wins
_RULES: list = [
    _CorrosionByColour(FailureType.CORROSION_PITTING,
                       "Orange-rust hue ({hue:.0f}°) and saturation ({sat:.0f}) indicate corrosion"),
    _CorrosionBySource(FailureType.CORROSION_PITTING,
                       "Colour-anomaly detection source indicates corrosion/rust region"),
    _ScreenCollapse(FailureType.SCREEN_COLLAPSE,
                    "Large area ({area:.1f}%) with low solidity ({sol:.2f}) indicates structural collapse"),
    _PluggingComplete(FailureType.PLUGGING_COMPLETE,
                      "Large ({area:.1f}%) dense fill ({ext:.2f}) indicates complete plugging"),
    _PluggingPartial(FailureType.PLUGGING_PARTIAL,
                     "High aspect ratio ({ar:.1f}) indicates a wide elongated band — partial plugging"),
    _WireWrapFailure(FailureType.WIRE_WRAP_FAILURE,
                     "Elongated ({ar:.1f}) low-circularity ({circ:.2f}) region suggests wire wrap damage"),
    _ErosionHole(FailureType.EROSION_HOLE,
                 "Compact dark blob (circ={circ:.2f}, ar={ar:.1f}) consistent with erosion hole"),
    _MechanicalDamage(FailureType.MECHANICAL_DAMAGE,
                      "No specific failure pattern matched — classified as mechanical damage"),
]


def classify_features(
    fv: FeatureVector,
    detection_index: int,
    class_name: str,
    severity: str,
    review_threshold: float = _REVIEW_THRESHOLD,
) -> ClassificationResult:
    """Apply ordered rules and return a ClassificationResult."""
    from .base import ClassificationResult  # avoid circular at module level

    for rule in _RULES:
        if not rule.matches(fv):
            continue
        conf = rule.confidence(fv)

        reasoning = rule.reasoning.format(
            hue=fv.mean_hue, sat=fv.mean_saturation, val=fv.mean_value,
            area=fv.area_pct, sol=fv.solidity, ext=fv.extent,
            ar=fv.aspect_ratio, circ=fv.circularity,
        )

        return ClassificationResult(
            detection_index=detection_index,
            class_name=class_name,
            failure_type=rule.failure_type,
            confidence=conf,
            severity=severity,
            requires_human_review=conf < review_threshold,
            reasoning=reasoning,
            features=fv,
        )

    # Absolute fallback
    return ClassificationResult(
        detection_index=detection_index,
        class_name=class_name,
        failure_type=FailureType.UNKNOWN,
        confidence=0.0,
        severity=severity,
        requires_human_review=True,
        reasoning="No rule matched — flagged for manual review",
        features=fv,
    )
