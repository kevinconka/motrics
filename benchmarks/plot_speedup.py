#!/usr/bin/env python3
"""Regenerate the README benchmark chart (light + dark SVGs).

Numbers below are wall time (ms) summed over all MOT17 sequences, taken from a
live CI run (see the benchmark comment on any PR). Update DATA after a fresh
`benchmarks/benchmark.py` run if the numbers drift, then:

    uv sync --group plot
    uv run python benchmarks/plot_speedup.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

ASSETS_DIR = Path(__file__).parent / "assets"

DATA: list[tuple[str, list[tuple[str, int]]]] = [
    ("CLEAR + Identity + HOTA", [("motrics", 770), ("TrackEval", 5930)]),
    ("CLEAR + Identity", [("motrics", 443), ("py-motmetrics", 6211)]),
]

# One accent color per mode (dataviz's validated categorical slot 1, blue) —
# every bar shares it, matching uv's benchmark chart: identity comes from the
# bold "motrics" label, not from a color-per-tool encoding. No surface color
# either; charts render on a transparent background.
PALETTES: dict[str, dict[str, str]] = {
    "light": {
        "accent": "#2a78d6",
        "primary": "#0b0b0b",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
    },
    "dark": {
        "accent": "#3987e5",
        "primary": "#ffffff",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
    },
}

BAR_HEIGHT = 0.6
ROW_STEP = 0.95
HEADING_STEP = 0.75
GROUP_GAP = 0.45


def _rounded_bar(ax: Axes, y: float, width: float, color: str) -> None:
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0, y - BAR_HEIGHT / 2),
            width,
            BAR_HEIGHT,
            boxstyle=f"round,pad=0,rounding_size={BAR_HEIGHT / 2}",
            linewidth=0,
            facecolor=color,
            mutation_aspect=1,
        )
    )


def render(mode: Literal["light", "dark"]) -> Path:
    palette = PALETTES[mode]
    max_ms = max(ms for _, bars in DATA for _, ms in bars)

    fig, ax = plt.subplots(figsize=(6.2, 3.0))

    yticks: list[float] = []
    yticklabels: list[str] = []
    label_weights: list[str] = []

    y = 0.0
    for label, bars in DATA:
        ax.text(0, y, label, va="center", ha="left", fontsize=9, color=palette["muted"])
        y -= HEADING_STEP

        for name, ms in bars:
            weight = "bold" if name == "motrics" else "normal"
            _rounded_bar(ax, y, ms, palette["accent"])
            ax.text(
                ms + max_ms * 0.02,
                y,
                f"{ms:,} ms",
                va="center",
                ha="left",
                fontsize=9,
                fontweight=weight,
                color=palette["primary"],
            )
            yticks.append(y)
            yticklabels.append(name)
            label_weights.append(weight)
            y -= ROW_STEP
        y -= GROUP_GAP

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=9, color=palette["primary"])
    for tick_label, weight in zip(ax.get_yticklabels(), label_weights, strict=True):
        tick_label.set_fontweight(weight)
    ax.tick_params(axis="y", length=0)

    ax.set_xlim(0, max_ms * 1.3)
    ax.set_ylim(y + GROUP_GAP - 0.3, 0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(palette["axis"])
    ax.tick_params(axis="x", colors=palette["muted"], labelsize=8)
    ax.set_xlabel(
        "wall time, ms (lower is better)", color=palette["muted"], fontsize=8.5
    )
    ax.grid(axis="x", color=palette["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout(pad=1.0)
    ASSETS_DIR.mkdir(exist_ok=True)
    out_path = ASSETS_DIR / f"speedup-{mode}.svg"
    fig.savefig(out_path, transparent=True)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    for mode in ("light", "dark"):
        path = render(mode)
        print(f"wrote {path}")
