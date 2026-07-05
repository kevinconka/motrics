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

# Validated via dataviz's scripts/validate_palette.js — see references/palette.md
# categorical slots 1 (blue), 2 (aqua), 3 (yellow). No surface color: charts
# render on a transparent background, so only mark/text colors are themed.
PALETTES: dict[str, dict[str, str]] = {
    "light": {
        "motrics": "#2a78d6",
        "TrackEval": "#1baf7a",
        "py-motmetrics": "#eda100",
        "primary": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
    },
    "dark": {
        "motrics": "#3987e5",
        "TrackEval": "#199e70",
        "py-motmetrics": "#c98500",
        "primary": "#ffffff",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
    },
}

BAR_HEIGHT = 0.6
ROW_STEP = 1.0
HEADING_STEP = 0.85
GROUP_GAP = 0.5


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

    fig, ax = plt.subplots(figsize=(6.4, 2.8))

    y = 0.0
    for label, bars in DATA:
        motrics_ms = bars[0][1]
        other_ms = bars[1][1]
        speedup = other_ms / motrics_ms
        ax.text(
            0,
            y,
            f"{label}  —  {speedup:.1f}x faster",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color=palette["secondary"],
        )
        y -= HEADING_STEP

        for name, ms in bars:
            _rounded_bar(ax, y, ms, palette[name])
            ax.text(
                ms + max_ms * 0.02,
                y,
                f"{name} · {ms:,} ms",
                va="center",
                ha="left",
                fontsize=9,
                color=palette["primary"],
            )
            y -= ROW_STEP
        y -= GROUP_GAP

    ax.set_xlim(0, max_ms * 1.38)
    ax.set_ylim(y + GROUP_GAP - 0.3, 0.6)
    ax.set_yticks([])
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
