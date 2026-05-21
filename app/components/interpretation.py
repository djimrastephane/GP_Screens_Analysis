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

_CLASSIFICATION_EVIDENCE: dict[str, str] = {
    "erosion_hole": (
        "Compact, dark regions with irregular or scalloped boundaries and low circularity "
        "(typically < 0.5), consistent with material loss through the screen aperture. "
        "Often co-located with areas of high defect density and reduced wire visibility. "
        "Low mean brightness in the defect region indicates absence of backing material."
    ),
    "corrosion_pitting": (
        "Clustered pits with roughly circular or oval morphology and irregular edge degradation. "
        "Surface-attack pattern typically spans multiple wires with uniform depth across the affected zone. "
        "High saturation contrast between corroded metal and surrounding wire surface. "
        "Pit distribution distinguishes chemical attack from mechanically-induced damage."
    ),
    "wire_wrap_failure": (
        "Linear or segmented discontinuities along the wire wrap path, with edge displacement "
        "inconsistent with the surrounding screen geometry. High aspect ratio (elongated shape) "
        "and alignment with the screen wire direction are key discriminators from circular pit damage. "
        "May show partial wire separation or distortion of the wrap pattern."
    ),
    "mechanical_damage": (
        "Large, irregular deformation zones with high solidity and aspect ratios inconsistent "
        "with corrosion or erosion patterns. Damage boundaries are typically abrupt rather than "
        "progressive. Crushed or bent wire geometry and directional deformation suggest impact "
        "loading from a specific direction during running, retrieval, or intervention."
    ),
    "plugging_partial": (
        "Low-contrast, diffuse regions with reduced wire visibility and elevated mean brightness "
        "relative to open screen areas. Texture analysis shows reduced spatial frequency "
        "(loss of sharp wire edges) consistent with surface deposition rather than material removal. "
        "Distribution is typically patchy rather than localised."
    ),
    "plugging_complete": (
        "Extensive low-contrast regions covering large contiguous areas with near-total loss "
        "of wire-wrap pattern visibility. Elevated and uniform brightness across the screen "
        "face is consistent with a continuous deposition layer obscuring the screen structure."
    ),
    "screen_collapse": (
        "Large-scale geometric distortion of the screen profile, with wire wrap spacing "
        "inconsistent with design geometry. High-solidity, large-area defect regions with "
        "internal structure indicating physical deformation of the screen body rather than "
        "localised material loss or deposition."
    ),
    "unknown": (
        "Detected region does not match the morphological signature of any classified failure type "
        "with sufficient confidence. The region may represent a novel failure mode, poor image "
        "quality in the area of interest, or a mixed mechanism requiring physical inspection."
    ),
}

_POTENTIAL_CAUSES: dict[frozenset, list[str]] = {
    frozenset({"erosion_hole", "corrosion_pitting"}): [
        "Erosion-corrosion: combined mechanical and chemical attack from abrasive sand in corrosive fluids",
        "High sand production velocity with concurrent corrosive produced water",
        "Localised high-velocity flow channelling through the gravel pack",
    ],
    frozenset({"erosion_hole"}): [
        "High-velocity abrasive sand erosion, possibly from non-uniform flow distribution",
        "Gravel pack failure creating direct sand impingement on the screen",
        "Production rate exceeding screen erosion velocity threshold",
    ],
    frozenset({"corrosion_pitting"}): [
        "Electrochemical corrosion from CO₂, H₂S, or oxygen ingress",
        "Produced water with elevated chloride or acid content",
        "Insufficient or failed corrosion inhibitor programme",
        "Galvanic corrosion at dissimilar metal contacts",
    ],
    frozenset({"mechanical_damage"}): [
        "Mechanical impact during running, retrieval, or well intervention",
        "Wellbore dog-leg or deviation causing contact damage",
        "Overpull or torque event during completion or workover",
    ],
    frozenset({"wire_wrap_failure"}): [
        "Fatigue from cyclic loading: flow-induced vibration, slug flow, or ESP operation",
        "Corrosion-assisted stress cracking at wire wrap welds",
        "Mechanical overload from differential pressure spike or hydraulic hammer",
    ],
    frozenset({"plugging_partial", "plugging_complete"}): [
        "Formation fines migration relative to screen slot size",
        "Scale deposition: carbonate, sulphate, or silicate",
        "Asphaltene or wax deposition from pressure or temperature changes",
        "Gravel pack consolidation or void formation promoting fines invasion",
    ],
    frozenset({"screen_collapse"}): [
        "Differential pressure across screen exceeded design collapse rating",
        "External annular pressure from formation or gravel pack consolidation",
        "Mechanical loading from overpull or set-down during retrieval",
    ],
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


def classification_evidence(failure_type: str) -> str:
    """Return morphological evidence text explaining why this failure type was classified."""
    return _CLASSIFICATION_EVIDENCE.get(failure_type, _CLASSIFICATION_EVIDENCE["unknown"])


def potential_causes(failure_types: list[str]) -> list[str]:
    """Return potential root causes for the observed combination of failure types."""
    ft_set = frozenset(failure_types)
    # Exact match first
    if ft_set in _POTENTIAL_CAUSES:
        return _POTENTIAL_CAUSES[ft_set]
    # Fall back to best partial match (most overlapping types)
    best, best_score = [], 0
    for key, causes in _POTENTIAL_CAUSES.items():
        score = len(ft_set & key)
        if score > best_score:
            best, best_score = causes, score
    if best:
        return best
    return ["Failure mechanism could not be determined from available failure types alone. "
            "Physical inspection and production history review recommended."]


def campaign_assessment(df) -> dict:
    """Synthesise a campaign-level engineering assessment from the summary dataframe."""
    import pandas as pd

    n_images = len(df)
    sev_counts = df["overall_severity"].value_counts().to_dict()
    overall_risk = "low"
    for level in ("critical", "high", "medium", "low"):
        if sev_counts.get(level, 0) > 0:
            overall_risk = level
            break

    # Failure type prevalence across images
    ft_image_counts: dict[str, int] = {}
    ft_defect_counts: dict[str, int] = {}
    import json
    for _, row in df.iterrows():
        breakdown = json.loads(row.get("failure_type_breakdown_json") or "{}")
        for ft, stats in breakdown.items():
            ft_image_counts[ft] = ft_image_counts.get(ft, 0) + 1
            ft_defect_counts[ft] = ft_defect_counts.get(ft, 0) + stats.get("count", 0)

    # Observed conditions narrative
    observed: list[str] = []
    sorted_fts = sorted(ft_image_counts, key=ft_image_counts.get, reverse=True)
    for ft in sorted_fts:
        n_img = ft_image_counts[ft]
        n_def = ft_defect_counts[ft]
        pct_imgs = n_img / n_images * 100
        label = ft.replace("_", " ").title()
        observed.append(
            f"{label} detected in {n_img} of {n_images} images ({pct_imgs:.0f}%), "
            f"{n_def} total detection{'s' if n_def != 1 else ''}."
        )

    mean_erosion = float(df["erosion_pct"].mean())
    max_erosion = float(df["erosion_pct"].max())
    max_erosion_file = df.loc[df["erosion_pct"].idxmax(), "source_filename"]
    observed.append(
        f"Mean campaign erosion: {mean_erosion:.1f}%. "
        f"Worst case: {max_erosion:.1f}% ({max_erosion_file})."
    )

    n_review = int(df["n_requires_review"].sum())
    if n_review > 0:
        observed.append(
            f"{n_review} detection{'s' if n_review != 1 else ''} flagged for human review "
            "due to classifier confidence below threshold."
        )

    # Recommended actions (synthesised from per-type recommendations)
    seen_actions: list[str] = []
    for ft in sorted_fts:
        narrative = _FAILURE_NARRATIVES.get(ft, _FAILURE_NARRATIVES["unknown"])
        for action in narrative["actions"]:
            if action not in seen_actions:
                seen_actions.append(action)

    # Risk description
    risk_descriptions = {
        "critical": "CRITICAL — one or more screens show catastrophic damage. Immediate intervention required.",
        "high": "HIGH — significant damage observed across multiple screens. Engineering review required.",
        "medium": "MODERATE — damage is developing. Monitor sand production and plan inspection.",
        "low": "LOW — minor damage observed. Continue scheduled monitoring.",
    }

    return {
        "risk": _RISK_LABEL.get(overall_risk, overall_risk.upper()),
        "risk_description": risk_descriptions.get(overall_risk, ""),
        "n_images": n_images,
        "sev_counts": sev_counts,
        "observed_conditions": observed,
        "failure_types_present": sorted_fts,
        "potential_causes": potential_causes(sorted_fts),
        "recommended_actions": seen_actions,
        "mean_erosion": mean_erosion,
        "max_erosion": max_erosion,
    }


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
