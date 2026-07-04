"""Sequences for the parity tests and the benchmark.

`make_synthetic()` builds deterministic in-memory sequences (used by the parity
tests); `load_real()` reads real MOTChallenge sequences fetched by `download.py`
(used by the benchmark).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import motrics
from motrics._types import Bbox

REAL_DIR = Path(__file__).parent / "data" / "real"


@dataclass(frozen=True)
class Sequence:
    """Frame-aligned ground-truth and predictions, ready for the metric functions."""

    name: str
    gt_ids: list[list[int]]
    gt_boxes: list[list[Bbox]]
    pred_ids: list[list[int]]
    pred_boxes: list[list[Bbox]]

    @property
    def num_frames(self) -> int:
        return len(self.gt_ids)

    @property
    def num_gt_dets(self) -> int:
        return sum(len(f) for f in self.gt_ids)

    @property
    def num_pred_dets(self) -> int:
        return sum(len(f) for f in self.pred_ids)


# name, frames, objects, seed — one small, one medium, one large.
_SPECS = [
    ("synth-small", 30, 6, 1),
    ("synth-medium", 150, 12, 2),
    ("synth-large", 300, 15, 3),
]


def _make(name: str, n_frames: int, n_obj: int, seed: int) -> Sequence:
    """Objects drift right; the tracker mostly detects them with jitter, and now
    and then misses one, swaps an id, or emits a false positive."""
    rng = random.Random(seed)
    w = h = 40.0
    gt_ids, gt_boxes, pred_ids, pred_boxes = [], [], [], []
    for t in range(n_frames):
        gi, gb, pi, pb = [], [], [], []
        for o in range(n_obj):
            if rng.random() < 0.92:
                x, y = 30.0 * o + 1.5 * t, 50.0 + 5.0 * ((o + t) % 7)
                gi.append(o)
                gb.append((x, y, x + w, y + h))
                if rng.random() < 0.85:
                    jx, jy = rng.uniform(-6, 6), rng.uniform(-6, 6)
                    pi.append(o if rng.random() > 0.04 else o + 1000)
                    pb.append((x + jx, y + jy, x + jx + w, y + jy + h))
        if rng.random() < 0.25:
            fx, fy = rng.uniform(0, 600), rng.uniform(300, 400)
            pi.append(5000 + t)
            pb.append((fx, fy, fx + w, fy + h))
        gt_ids.append(gi)
        gt_boxes.append(gb)
        pred_ids.append(pi)
        pred_boxes.append(pb)
    return Sequence(name, gt_ids, gt_boxes, pred_ids, pred_boxes)


def make_synthetic() -> list[Sequence]:
    """Deterministic synthetic sequences, generated in memory."""
    return [_make(*spec) for spec in _SPECS]


def load_real() -> list[Sequence]:
    """Real sequences from ``data/real/<seq>/`` (see download.py).

    Ground truth is filtered through ``preprocess_motchallenge`` (distractor
    removal, pedestrian-only, "do not consider" rows dropped) so these numbers
    reflect what TrackEval actually evaluates, not raw box IoU.
    """
    if not REAL_DIR.exists():
        return []
    seqs = []
    for d in sorted(p for p in REAL_DIR.iterdir() if p.is_dir()):
        gt_file, pred_file = d / "gt" / "gt.txt", d / "pred.txt"
        if gt_file.is_file() and pred_file.is_file():
            gt = motrics.load_motchallenge_gt(gt_file)
            pred = motrics.load_motchallenge(pred_file)
            benchmark = d.name.split("-")[0]  # e.g. "MOT17" from "MOT17-02-FRCNN"
            args = motrics.preprocess_motchallenge(gt, pred, benchmark=benchmark)
            seqs.append(Sequence(d.name, *args))
    return seqs
