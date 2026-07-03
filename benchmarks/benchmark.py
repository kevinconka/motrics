#!/usr/bin/env python3
"""Benchmark motrics against TrackEval and py-motmetrics on real MOTChallenge data.

CLEAR / Identity / HOTA are run through each engine, checked for agreement
(a parity gate), and timed. Requires the parity group and downloaded data:

    uv sync --group parity
    uv run maturin develop --release --uv    # release! debug is ~10x slower
    uv run python benchmarks/download.py
    uv run python benchmarks/benchmark.py [--repeats N] [--smoke]
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from typing import Any

import motmetrics as mm
import motrics
import numpy as np
from fixtures import Sequence, load_real
from trackeval.metrics import CLEAR, HOTA, Identity

IOU_THRESHOLD = 0.5
# On dense real data, Hungarian ties are resolved differently by each solver, so
# a couple of matches/switches shift and metrics differ by ~1e-4. That is not a
# bug — tests/test_parity.py enforces exact 1e-9 parity on tie-free synthetic
# data; this gate only catches real numeric divergence.
PARITY_ATOL = 5e-3


def _time(fn: Callable[[], Any], repeats: int) -> tuple[Any, float]:
    """Run ``fn`` ``repeats`` times; return its result and the best wall time (s)."""
    best = float("inf")
    result: Any = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - start)
    return result, best


def _np_iou(gt_boxes: list, pred_boxes: list):  # -> np.ndarray
    """Vectorised IoU matrix, used by the TrackEval and py-motmetrics paths."""
    if not gt_boxes or not pred_boxes:
        return np.zeros((len(gt_boxes), len(pred_boxes)), dtype=float)
    g, p = np.asarray(gt_boxes, dtype=float), np.asarray(pred_boxes, dtype=float)
    x1 = np.maximum(g[:, None, 0], p[None, :, 0])
    y1 = np.maximum(g[:, None, 1], p[None, :, 1])
    x2 = np.minimum(g[:, None, 2], p[None, :, 2])
    y2 = np.minimum(g[:, None, 3], p[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    p_area = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1])
    union = area[:, None] + p_area[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def _motrics_clear(seq: Sequence) -> dict[str, float]:
    m = motrics.compute_clear(*_args(seq), IOU_THRESHOLD)
    return {"MOTA": m.mota, "MOTP": m.motp}


def _motrics_identity(seq: Sequence) -> dict[str, float]:
    return {"IDF1": motrics.compute_identity(*_args(seq), IOU_THRESHOLD).idf1}


def _motrics_hota(seq: Sequence) -> dict[str, float]:
    m = motrics.compute_hota(*_args(seq))
    return {"HOTA": m.hota, "DetA": m.deta, "AssA": m.assa}


def _args(seq: Sequence) -> tuple:
    return seq.gt_ids, seq.gt_boxes, seq.pred_ids, seq.pred_boxes


def _contiguous(frames: list[list[int]]) -> list:
    """Remap ids to contiguous 0..n integers per frame (what TrackEval expects)."""
    mapping = {v: k for k, v in enumerate(sorted({i for f in frames for i in f}))}
    return [np.array([mapping[i] for i in f], dtype=int) for f in frames]


def _trackeval_all(seq: Sequence) -> dict[str, float]:
    boxes = zip(seq.gt_boxes, seq.pred_boxes, strict=True)
    data = {
        "num_timesteps": seq.num_frames,
        "num_gt_ids": len({i for f in seq.gt_ids for i in f}),
        "num_tracker_ids": len({i for f in seq.pred_ids for i in f}),
        "num_gt_dets": seq.num_gt_dets,
        "num_tracker_dets": seq.num_pred_dets,
        "gt_ids": _contiguous(seq.gt_ids),
        "tracker_ids": _contiguous(seq.pred_ids),
        "similarity_scores": [_np_iou(g, p) for g, p in boxes],
    }
    cfg = {"THRESHOLD": IOU_THRESHOLD, "PRINT_CONFIG": False}
    clear = CLEAR(cfg).eval_sequence(data)
    ident = Identity(cfg).eval_sequence(data)
    hota = HOTA({"PRINT_CONFIG": False}).eval_sequence(data)
    return {
        "MOTA": float(clear["MOTA"]),
        "MOTP": float(clear["MOTP"]),
        "IDF1": float(ident["IDF1"]),
        "HOTA": float(np.mean(hota["HOTA"])),
        "DetA": float(np.mean(hota["DetA"])),
        "AssA": float(np.mean(hota["AssA"])),
    }


def _motmetrics_all(seq: Sequence) -> dict[str, float]:
    acc = mm.MOTAccumulator(auto_id=True)
    frames = zip(seq.gt_boxes, seq.pred_boxes, seq.gt_ids, seq.pred_ids, strict=True)
    for g, p, gi, pi in frames:
        iou = _np_iou(g, p)
        # motmetrics wants 1 - IoU, with sub-threshold pairs masked out.
        acc.update(gi, pi, np.where(iou >= IOU_THRESHOLD, 1.0 - iou, np.nan))
    row = mm.metrics.create().compute(acc, metrics=["mota", "idf1"], name="seq").iloc[0]
    return {"MOTA": float(row["mota"]), "IDF1": float(row["idf1"])}


def _check_parity(results: dict[str, dict[str, float]]) -> list[str]:
    """Flag summary-metric disagreements beyond assignment tie-breaking noise."""
    base = results["motrics"]
    notes = []
    for engine, keys in (
        ("trackeval", ("MOTA", "MOTP", "IDF1", "HOTA", "DetA", "AssA")),
        ("motmetrics", ("MOTA", "IDF1")),
    ):
        for key in keys:
            other = results[engine].get(key)
            if other is not None and abs(base[key] - other) > PARITY_ATOL:
                notes.append(f"  ⚠ {key}: motrics={base[key]:.6f} {engine}={other:.6f}")
    return notes


def run(sequences: list[Sequence], repeats: int) -> int:
    if motrics.is_debug_build():
        print(
            "WARNING: debug build — timings are not meaningful (~10x slow). "
            "Rebuild: uv run maturin develop --release --uv"
        )
    print(f"\n{len(sequences)} sequence(s), {repeats} repeat(s)\n")

    motrics_all = motrics_ci = trackeval = motmetrics = 0.0
    ok = True
    for seq in sequences:
        c, t_c = _time(lambda s=seq: _motrics_clear(s), repeats)
        i, t_i = _time(lambda s=seq: _motrics_identity(s), repeats)
        h, t_h = _time(lambda s=seq: _motrics_hota(s), repeats)
        te, t_te = _time(lambda s=seq: _trackeval_all(s), repeats)
        mmr, t_mm = _time(lambda s=seq: _motmetrics_all(s), repeats)

        m = {**c, **i, **h}
        notes = _check_parity({"motrics": m, "trackeval": te, "motmetrics": mmr})
        ok = ok and not notes
        print(
            f"{seq.name}  {seq.num_frames} frames, "
            f"{seq.num_gt_dets} gt / {seq.num_pred_dets} pred  |  "
            f"MOTA={m['MOTA']:.3f} IDF1={m['IDF1']:.3f} HOTA={m['HOTA']:.3f}  |  "
            f"parity {'OK' if not notes else 'MISMATCH'}"
        )
        for note in notes:
            print(note)

        motrics_all += t_c + t_i + t_h
        motrics_ci += t_c + t_i
        trackeval += t_te
        motmetrics += t_mm

    print("\n" + "=" * 60)
    print(f"Speedup vs motrics over {len(sequences)} sequence(s) (higher = faster):")
    print(f"  TrackEval      {trackeval / motrics_all:.1f}x  (CLEAR + Identity + HOTA)")
    print(f"  py-motmetrics  {motmetrics / motrics_ci:.1f}x  (CLEAR + Identity)")
    print("=" * 60)

    if not ok:
        print("\nPARITY FAILURES DETECTED — see above.")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5, help="timing repeats")
    parser.add_argument("--smoke", action="store_true", help="single repeat")
    args = parser.parse_args()

    sequences = load_real()
    if not sequences:
        print("error: no sequences found. Run: uv run python benchmarks/download.py")
        return 1
    return run(sequences, 1 if args.smoke else args.repeats)


if __name__ == "__main__":
    raise SystemExit(main())
