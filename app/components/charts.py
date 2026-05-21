"""Plotly chart builders for the Streamlit dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Colour maps (matching annotation module)
_FAILURE_TYPE_COLOURS = {
    "erosion_hole":       "#DC3737",
    "corrosion_pitting":  "#E69B19",
    "wire_wrap_failure":  "#28B4DC",
    "screen_collapse":    "#AF23AF",
    "mechanical_damage":  "#5A6EDC",
    "plugging_partial":   "#2DC35F",
    "plugging_complete":  "#0F5F32",
    "unknown":            "#969696",
}
_SEVERITY_COLOURS = {
    "low":      "#33A633",
    "medium":   "#E09914",
    "high":     "#D94814",
    "critical": "#BF0D0D",
}

_PLOTLY_TEMPLATE = "plotly_white"


def erosion_bar(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar: erosion % per image, coloured by severity."""
    sorted_df = df.sort_values("erosion_pct", ascending=True)

    colours = [_SEVERITY_COLOURS.get(s, "#888") for s in sorted_df["overall_severity"]]

    fig = go.Figure(go.Bar(
        x=sorted_df["erosion_pct"],
        y=sorted_df["source_filename"],
        orientation="h",
        marker_color=colours,
        text=[f"{v:.1f}%" for v in sorted_df["erosion_pct"]],
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Erosion: %{x:.1f}%<br>"
            "<extra></extra>"
        ),
    ))
    fig.add_vline(x=20, line_dash="dash", line_color="#aaa", line_width=1,
                  annotation_text="20%", annotation_position="top right")
    fig.add_vline(x=50, line_dash="dash", line_color="#cc4400", line_width=1,
                  annotation_text="50%", annotation_position="top right")
    fig.update_layout(
        template=_PLOTLY_TEMPLATE,
        xaxis_title="Erosion % (screen content area)",
        xaxis=dict(range=[0, max(df["erosion_pct"]) * 1.2 if len(df) else 50]),
        yaxis_title="",
        height=max(280, len(df) * 40),
        margin=dict(l=10, r=80, t=10, b=40),
        showlegend=False,
    )
    return fig


def failure_type_bar(type_counts: dict[str, int]) -> go.Figure:
    """Horizontal bar: failure type counts, colour-coded by type."""
    if not type_counts:
        return go.Figure()

    sorted_types = sorted(type_counts.items(), key=lambda x: -x[1])
    labels = [t.replace("_", " ").title() for t, _ in sorted_types]
    counts = [c for _, c in sorted_types]
    raw_types = [t for t, _ in sorted_types]
    colours = [_FAILURE_TYPE_COLOURS.get(t, "#888") for t in raw_types]

    fig = go.Figure(go.Bar(
        x=counts,
        y=labels,
        orientation="h",
        marker_color=colours,
        text=counts,
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Count: %{x}<extra></extra>",
    ))
    fig.update_layout(
        template=_PLOTLY_TEMPLATE,
        xaxis_title="Total detections",
        yaxis_title="",
        height=max(220, len(type_counts) * 38),
        margin=dict(l=10, r=60, t=10, b=30),
        showlegend=False,
    )
    return fig


def severity_pie(severity_counts: dict[str, int]) -> go.Figure:
    """Donut chart: severity distribution."""
    if not severity_counts:
        return go.Figure()

    order = ["low", "medium", "high", "critical"]
    labels, values, colours = [], [], []
    for sev in order:
        if sev in severity_counts and severity_counts[sev] > 0:
            labels.append(sev.capitalize())
            values.append(severity_counts[sev])
            colours.append(_SEVERITY_COLOURS[sev])

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker_colors=colours,
        hole=0.45,
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>%{value} images<extra></extra>",
    ))
    fig.update_layout(
        template=_PLOTLY_TEMPLATE,
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5),
        margin=dict(l=10, r=10, t=10, b=10),
        height=260,
    )
    return fig


def scatter_erosion_defects(df: pd.DataFrame) -> go.Figure:
    """Scatter: erosion % vs defect count per image."""
    colours = [_SEVERITY_COLOURS.get(s, "#888") for s in df["overall_severity"]]

    fig = go.Figure(go.Scatter(
        x=df["erosion_pct"],
        y=df["n_defects"],
        mode="markers+text",
        text=df["source_filename"].str[:12],
        textposition="top center",
        marker=dict(size=12, color=colours, line=dict(color="white", width=1)),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Erosion: %{x:.1f}%<br>"
            "Defects: %{y}<extra></extra>"
        ),
    ))
    fig.update_layout(
        template=_PLOTLY_TEMPLATE,
        xaxis_title="Erosion %",
        yaxis_title="Defect Count",
        height=350,
        margin=dict(l=10, r=10, t=10, b=40),
    )
    return fig


def failure_type_breakdown_pie(breakdown: dict) -> go.Figure:
    """Pie for one image's failure type breakdown."""
    if not breakdown:
        return go.Figure()

    labels = [k.replace("_", " ").title() for k in breakdown]
    values = [v["count"] for v in breakdown.values()]
    colours = [_FAILURE_TYPE_COLOURS.get(k, "#888") for k in breakdown]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker_colors=colours,
        hole=0.40,
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<extra></extra>",
    ))
    fig.update_layout(
        template=_PLOTLY_TEMPLATE,
        showlegend=False,
        margin=dict(l=5, r=5, t=5, b=5),
        height=220,
    )
    return fig
