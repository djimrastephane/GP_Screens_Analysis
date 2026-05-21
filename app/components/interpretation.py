"""Auto-generate engineering interpretation from computer vision pipeline results."""

from __future__ import annotations

_FAILURE_NARRATIVES: dict[str, dict] = {
    "erosion_hole": {
        "mechanism": "Erosion",
        "high_critical": (
            "Erosion holes detected across {pct:.0f}% of visible screen area. "
            "Produced sand has breached the screen membrane, creating direct communication "
            "between formation and wellbore. Screen flow competency is compromised."
        ),
        "medium": (
            "Localised erosion holes detected across {pct:.0f}% of visible screen area. "
            "Early-stage membrane breach is present; monitor sand production rates closely."
        ),
        "low": "Minor erosion pitting detected. Screen retains structural integrity.",
        "actions": [
            "Review cumulative sand production records against screen design limits",
            "Assess production velocity — erosion rate scales with flow velocity cubed",
            "Consider re-completion or screen replacement if sand rate is increasing",
        ],
    },
    "corrosion_pitting": {
        "mechanism": "Corrosion",
        "high_critical": (
            "Extensive corrosion pitting across {pct:.0f}% of visible screen area. "
            "Pattern suggests prolonged exposure to corrosive fluids or inadequate "
            "corrosion inhibition. Metallurgical analysis is recommended."
        ),
        "medium": (
            "Moderate corrosion pitting observed across {pct:.0f}% of screen area. "
            "Corrosion inhibitor programme and produced fluid chemistry should be assessed."
        ),
        "low": "Minor corrosion pitting detected. Continue monitoring fluid chemistry.",
        "actions": [
            "Review corrosion inhibitor injection rates and programme effectiveness",
            "Assess produced water chemistry: pH, CO₂, H₂S partial pressures",
            "Recommend metallurgical analysis to confirm corrosion mechanism",
            "Check for galvanic corrosion at dissimilar metal contacts",
        ],
    },
    "wire_wrap_failure": {
        "mechanism": "Mechanical / Fatigue",
        "high_critical": (
            "Wire-wrap integrity significantly compromised across {pct:.0f}% of screen area. "
            "Failure may indicate mechanical overload, vibration fatigue, or corrosion-assisted "
            "cracking during service or retrieval."
        ),
        "medium": (
            "Partial wire-wrap failure detected. Structural integrity is reduced; "
            "further damage likely under continued production."
        ),
        "low": "Minor wire-wrap deformation observed. Monitor for progression.",
        "actions": [
            "Review running and retrieval records for over-pull or torque events",
            "Check for vibration sources: ESP proximity, gas slugging, choke cycling",
            "Review screen specification against actual differential pressure and flow rates",
        ],
    },
    "mechanical_damage": {
        "mechanism": "Mechanical Impact",
        "high_critical": (
            "Significant mechanical damage across {pct:.0f}% of screen area. "
            "Impact or abrasion likely occurred during running, retrieval, or a well intervention."
        ),
        "medium": "Moderate mechanical damage detected. Review operational history.",
        "low": "Minor mechanical damage observed.",
        "actions": [
            "Review BHA and tool selection for recent interventions",
            "Check wellbore deviation and dog-leg severity at screen depth",
            "Review running speed and pick-up / set-down weights from completion report",
        ],
    },
    "plugging_partial": {
        "mechanism": "Plugging",
        "high_critical": (
            "Extensive screen plugging across {pct:.0f}% of visible area. "
            "Significant restriction to inflow is present. "
            "Possible causes: formation fines migration, scale, asphaltene, or wax deposition."
        ),
        "medium": (
            "Partial screen plugging detected across {pct:.0f}% of visible area. "
            "Inflow restriction is developing."
        ),
        "low": "Minor plugging observed. Monitor productivity index trend.",
        "actions": [
            "Assess formation fines migration relative to screen slot size",
            "Test for scale, asphaltene, or wax based on produced fluid chemistry",
            "Consider matrix stimulation or solvent squeeze if deposition is confirmed",
            "Review gravel pack integrity — plugging sometimes indicates pack voids",
        ],
    },
    "plugging_complete": {
        "mechanism": "Complete Plugging",
        "high_critical": (
            "Screen fully plugged — no inflow path remains through the screen. "
            "Immediate intervention is required."
        ),
        "medium": "Near-complete plugging detected. Production will be severely impaired.",
        "low": "Early-stage plugging detected.",
        "actions": [
            "Investigate workaround options: re-perforation above/below the screen",
            "Perform fluid analysis to identify the plugging agent before treatment",
            "Consider chemical treatment or re-completion",
        ],
    },
    "screen_collapse": {
        "mechanism": "Structural Collapse",
        "high_critical": (
            "Screen collapse detected. Structural integrity has failed. "
            "Differential pressure or mechanical loading exceeded the screen design rating."
        ),
        "medium": "Partial screen collapse detected. Do not re-run without engineering review.",
        "low": "Early-stage deformation observed — monitor for progression.",
        "actions": [
            "Do not re-run this screen — full replacement is required",
            "Review gravel pack consolidation and differential pressure history",
            "Check for annular collapse pressure anomalies at screen depth",
        ],
    },
    "unknown": {
        "mechanism": "Unclassified",
        "high_critical": (
            "Significant damage across {pct:.0f}% of screen area. "
            "Failure mechanism could not be classified with sufficient confidence."
        ),
        "medium": "Damage of unclassified mechanism detected.",
        "low": "Minor unclassified damage observed.",
        "actions": [
            "Recommend physical inspection by a completions or integrity specialist",
            "Submit representative sample for metallurgical analysis if available",
        ],
    },
}

_RISK_LABEL = {"low": "LOW", "medium": "MODERATE", "high": "HIGH", "critical": "CRITICAL"}

_SEVERITY_BASIS = (
    "Severity is based on total defect area as a percentage of visible screen area: "
    "< 5 % → Low  |  5–20 % → Medium  |  "
    "20–50 % → High  |  ≥ 50 % → Critical. "
    "Screen collapse and complete plugging escalate one severity level regardless of area, "
    "as they represent total functional failure. "
    "Thresholds are engineering defaults; calibration against historical failure data "
    "and screen manufacturer limits is recommended."
)

EROSION_PCT_DEFINITION = (
    "Erosion % = total defect pixel area ÷ visible screen pixel area × 100. "
    "Measured on the visible screen content only — letterbox padding added during "
    "preprocessing is excluded. This is a model estimate of damaged area fraction, "
    "not a direct measurement of metal loss or open-flow area increase."
)


def engineering_interpretation(
    dominant_failure_type: str,
    erosion_pct: float,
    severity: str,
    n_requires_review: int,
    mean_confidence: float | None,
) -> dict:
    """Return a structured engineering interpretation for display in the dashboard."""
    ft = dominant_failure_type or "unknown"
    narrative = _FAILURE_NARRATIVES.get(ft, _FAILURE_NARRATIVES["unknown"])

    if severity in ("high", "critical"):
        tmpl = narrative["high_critical"]
    elif severity == "medium":
        tmpl = narrative["medium"]
    else:
        tmpl = narrative["low"]

    try:
        detail = tmpl.format(pct=erosion_pct)
    except KeyError:
        detail = tmpl

    actions = list(narrative["actions"])
    if n_requires_review > 0:
        actions.append(
            f"{n_requires_review} detection(s) below confidence threshold — "
            "verify flagged regions before using results in engineering decisions."
        )

    return {
        "risk": _RISK_LABEL.get(severity, severity.upper()),
        "mechanism": narrative["mechanism"],
        "detail": detail,
        "actions": actions,
        "confidence_note": (
            f"Mean model confidence: {mean_confidence:.0%}" if mean_confidence is not None else None
        ),
        "severity_basis": _SEVERITY_BASIS,
    }
