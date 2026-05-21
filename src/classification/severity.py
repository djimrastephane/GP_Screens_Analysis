"""Severity scoring from area percentage and failure type."""

from __future__ import annotations

from pathlib import Path

import yaml

_SEVERITY_ORDER = ["low", "medium", "high", "critical"]

_DEFAULT_THRESHOLDS = {
    "low": 5.0,
    "medium": 20.0,
    "high": 50.0,
    "critical": 100.0,
}

# Failure types that escalate severity by one level regardless of area
_ESCALATING_TYPES = {"screen_collapse", "plugging_complete"}


def load_thresholds(config_path: Path | None = None) -> dict[str, float]:
    if config_path and config_path.exists():
        with config_path.open() as f:
            raw = yaml.safe_load(f)
        return raw.get("severity_thresholds", _DEFAULT_THRESHOLDS)
    return _DEFAULT_THRESHOLDS


def area_to_severity(
    area_pct: float,
    failure_type: str = "",
    thresholds: dict[str, float] | None = None,
) -> str:
    """Map defect area percentage to a severity label.

    Screen collapse and complete plugging escalate one level regardless of area
    because they represent total functional failure.
    """
    t = thresholds or _DEFAULT_THRESHOLDS

    if area_pct < t["low"]:
        base = "low"
    elif area_pct < t["medium"]:
        base = "medium"
    elif area_pct < t["high"]:
        base = "high"
    else:
        base = "critical"

    if failure_type in _ESCALATING_TYPES:
        idx = _SEVERITY_ORDER.index(base)
        base = _SEVERITY_ORDER[min(idx + 1, len(_SEVERITY_ORDER) - 1)]

    return base


def worst_severity(severities: list[str]) -> str:
    """Return the highest severity from a list."""
    if not severities:
        return "low"
    return max(severities, key=lambda s: _SEVERITY_ORDER.index(s))
