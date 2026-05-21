"""Matplotlib chart generation for PDF embedding."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from .styles import FAILURE_TYPE_HEX, SEVERITY_COL


# Map severity label to hex for matplotlib
_SEV_HEX = {
    "low":      "#33A633",
    "medium":   "#E09914",
    "high":     "#D94814",
    "critical": "#BF0D0D",
}


def erosion_bar_chart(
    filenames: list[str],
    erosion_pcts: list[float],
    severities: list[str],
    output_path: Path,
    figsize: tuple[float, float] = (8, 4),
) -> Path:
    """Horizontal bar chart: erosion % per image, coloured by severity."""
    n = len(filenames)
    fig, ax = plt.subplots(figsize=figsize)

    bar_colours = [_SEV_HEX.get(s, "#969696") for s in severities]
    short_names = [f[:22] + "…" if len(f) > 22 else f for f in filenames]

    y_pos = np.arange(n)
    bars = ax.barh(y_pos, erosion_pcts, color=bar_colours, edgecolor="white", linewidth=0.5)

    # Value labels
    for bar, val in zip(bars, erosion_pcts):
        ax.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", ha="left", fontsize=8, color="#333333",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel("Erosion % (screen region)", fontsize=9)
    ax.set_title("Erosion % by Image", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlim(0, max(erosion_pcts) * 1.18 if erosion_pcts else 10)
    ax.axvline(x=20, color="#999", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(x=50, color="#cc4400", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()

    # Legend patches
    legend_handles = [
        mpatches.Patch(color=_SEV_HEX[s], label=s.capitalize())
        for s in ["low", "medium", "high", "critical"]
        if s in severities
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=7, loc="lower right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def failure_type_chart(
    type_counts: dict[str, int],
    output_path: Path,
    figsize: tuple[float, float] = (7, 3.5),
) -> Path:
    """Horizontal bar chart for failure type counts across all images."""
    if not type_counts:
        return output_path

    types = list(type_counts.keys())
    counts = [type_counts[t] for t in types]
    colours = [FAILURE_TYPE_HEX.get(t, "#969696") for t in types]
    short_types = [t.replace("_", " ") for t in types]

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(types))
    bars = ax.barh(y_pos, counts, color=colours, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, counts):
        ax.text(
            bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", ha="left", fontsize=8,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_types, fontsize=8)
    ax.set_xlabel("Total detections", fontsize=9)
    ax.set_title("Failure Type Distribution", fontsize=11, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def severity_pie_chart(
    severity_counts: dict[str, int],
    output_path: Path,
    figsize: tuple[float, float] = (4, 4),
) -> Path:
    """Pie chart showing overall severity distribution."""
    if not severity_counts:
        return output_path

    order = ["low", "medium", "high", "critical"]
    labels, sizes, colours = [], [], []
    for sev in order:
        if sev in severity_counts and severity_counts[sev] > 0:
            labels.append(sev.capitalize())
            sizes.append(severity_counts[sev])
            colours.append(_SEV_HEX[sev])

    if not sizes:
        return output_path

    fig, ax = plt.subplots(figsize=figsize)
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colours,
        autopct="%1.0f%%", startangle=140,
        textprops={"fontsize": 9},
        pctdistance=0.82,
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax.set_title("Severity Distribution", fontsize=11, fontweight="bold", pad=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
