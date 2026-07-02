#!/usr/bin/env python3
"""Generate the shared MOTChallenge-format fixtures used by parity and benchmarks.

The fixtures are small, synthetic-but-realistic tracking sequences written in
MOTChallenge CSV format (``frame, id, bb_left, bb_top, bb_width, bb_height,
conf, x, y, z``). Both the parity tests (``tests/test_parity.py``) and the
benchmark suite (``benchmark.py``) consume *one* shared set of inputs, loaded
through the public ``motrics.load_motchallenge`` reader.

Generation is deterministic (seeded ``random.Random``), so it reproduces
byte-identical files and is regenerated on demand — the output is git-ignored,
not committed:

    uv run python benchmarks/generate_fixtures.py

Real MOTChallenge sequences (fetched by ``download.py``) land under
``benchmarks/data/real/`` and take precedence in the benchmark; these synthetic
fixtures are the network-free fallback that keeps the suite reproducible
anywhere.
"""

from __future__ import annotations

import random
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# (name, num_frames, num_objects, seed) — one small, one medium, one large.
SEQUENCES: list[tuple[str, int, int, int]] = [
    ("MOTRICS-SYNTH-01", 30, 6, 1),
    ("MOTRICS-SYNTH-02", 150, 12, 2),
    ("MOTRICS-SYNTH-03", 500, 20, 3),
]


def _mot_line(
    frame: int, obj_id: int, box: tuple[float, float, float, float], conf: float
) -> str:
    """Format one MOTChallenge row from an ``xywh`` box (left, top, width, height)."""
    left, top, width, height = box
    return (
        f"{frame},{obj_id},{left:.2f},{top:.2f},{width:.2f},{height:.2f},"
        f"{conf:.4f},-1,-1,-1"
    )


def _generate(
    num_frames: int, num_objects: int, seed: int
) -> tuple[list[str], list[str]]:
    """Build (gt_lines, pred_lines) for one sequence.

    Objects drift left-to-right; the tracker mostly detects them with positional
    jitter, occasionally misses one, rarely swaps an id, and now and then emits a
    spurious false positive. This exercises TP/FP/FN, id switches and a spread of
    IoU values (so HOTA's alpha sweep sees non-trivial curves).
    """
    rng = random.Random(seed)
    gt_lines: list[str] = []
    pred_lines: list[str] = []
    w = h = 40.0

    for t in range(1, num_frames + 1):
        for o in range(num_objects):
            # Ground-truth object present most of the time.
            if rng.random() < 0.92:
                x = 30.0 * o + 1.5 * t
                y = 50.0 + 5.0 * ((o + t) % 7)
                gt_lines.append(_mot_line(t, o, (x, y, w, h), 1.0))
                # Tracker detects it most of the time, with jitter.
                if rng.random() < 0.85:
                    jx, jy = rng.uniform(-6, 6), rng.uniform(-6, 6)
                    # Rare identity switch (offset id keeps it distinct).
                    pid = o if rng.random() > 0.04 else o + 1000
                    conf = rng.uniform(0.5, 1.0)
                    pred_lines.append(_mot_line(t, pid, (x + jx, y + jy, w, h), conf))
        # Occasional spurious false positive.
        if rng.random() < 0.25:
            fx, fy = rng.uniform(0, 600), rng.uniform(300, 400)
            pred_lines.append(
                _mot_line(t, 5000 + t, (fx, fy, w, h), rng.uniform(0.5, 1.0))
            )

    return gt_lines, pred_lines


def main() -> None:
    for name, num_frames, num_objects, seed in SEQUENCES:
        gt_lines, pred_lines = _generate(num_frames, num_objects, seed)
        seq_dir = DATA_DIR / name
        (seq_dir / "gt").mkdir(parents=True, exist_ok=True)
        (seq_dir / "gt" / "gt.txt").write_text(
            "\n".join(gt_lines) + "\n", encoding="utf-8"
        )
        (seq_dir / "pred.txt").write_text(
            "\n".join(pred_lines) + "\n", encoding="utf-8"
        )
        print(
            f"{name}: {num_frames} frames, {num_objects} objects "
            f"-> {len(gt_lines)} gt / {len(pred_lines)} pred detections"
        )


if __name__ == "__main__":
    main()
