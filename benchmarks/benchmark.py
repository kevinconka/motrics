#!/usr/bin/env python3
"""Benchmark motrics against TrackEval and py-motmetrics.

For each sequence, CLEAR / Identity / HOTA are run through every available
engine, checked for agreement (a parity gate) and timed end-to-end. TrackEval
and py-motmetrics are optional (``uv sync --group parity``). Runs on the real
MOTChallenge sequences fetched by download.py. Build in release mode — a debug
build is ~10x slower.

    uv run python benchmarks/download.py     # once
    uv run python benchmarks/benchmark.py [--repeats N] [--smoke]
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from typing import Any

import motrics
from fixtures import Sequence, load_real

# --- optional reference engines -------------------------------------------------

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a bench/parity dependency
    np = None  # type: ignore[assignment]

try:
    from trackeval.metrics import CLEAR as TE_CLEAR
    from trackeval.metrics import HOTA as TE_HOTA
    from trackeval.metrics import Identity as TE_Identity

    _HAS_TRACKEVAL = np is not None
except ImportError:
    _HAS_TRACKEVAL = False

try:
    import motmetrics as mm

    _HAS_MOTMETRICS = np is not None
except ImportError:
    _HAS_MOTMETRICS = False

IOU_THRESHOLD = 0.5


# --- timing ---------------------------------------------------------------------


def _time(fn: Callable[[], Any], repeats: int) -> tuple[Any, float]:
    """Run ``fn`` ``repeats`` times; return its result and the best wall time (s)."""
    best = float("inf")
    result: Any = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - start)
    return result, best


# --- motrics --------------------------------------------------------------------


def _motrics_clear(seq: Sequence) -> dict[str, float]:
    m = motrics.compute_clear(
        seq.gt_ids, seq.gt_boxes, seq.pred_ids, seq.pred_boxes, IOU_THRESHOLD
    )
    return {"MOTA": m.mota, "MOTP": m.motp, "IDSW": float(m.num_switches)}


def _motrics_identity(seq: Sequence) -> dict[str, float]:
    m = motrics.compute_identity(
        seq.gt_ids, seq.gt_boxes, seq.pred_ids, seq.pred_boxes, IOU_THRESHOLD
    )
    return {"IDF1": m.idf1}


def _motrics_hota(seq: Sequence) -> dict[str, float]:
    m = motrics.compute_hota(seq.gt_ids, seq.gt_boxes, seq.pred_ids, seq.pred_boxes)
    return {"HOTA": m.hota, "DetA": m.deta, "AssA": m.assa}


# --- TrackEval ------------------------------------------------------------------


def _np_iou_matrix(gt_boxes: list, pred_boxes: list):  # -> np.ndarray
    """Vectorised IoU for the TrackEval path (its own, motrics-independent, IoU)."""
    if not gt_boxes or not pred_boxes:
        return np.zeros((len(gt_boxes), len(pred_boxes)), dtype=float)
    g = np.asarray(gt_boxes, dtype=float)
    p = np.asarray(pred_boxes, dtype=float)
    x1 = np.maximum(g[:, None, 0], p[None, :, 0])
    y1 = np.maximum(g[:, None, 1], p[None, :, 1])
    x2 = np.minimum(g[:, None, 2], p[None, :, 2])
    y2 = np.minimum(g[:, None, 3], p[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ga = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    pa = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1])
    union = ga[:, None] + pa[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def _trackeval_data(seq: Sequence, sims: list | None = None) -> dict:
    """Build TrackEval's per-sequence data dict (contiguous ids + similarities).

    ``sims`` lets the parity check inject identical similarities (from
    ``motrics.iou_matrix``); when ``None`` the timed path computes its own.
    """
    gt_id_set = sorted({i for f in seq.gt_ids for i in f})
    tr_id_set = sorted({i for f in seq.pred_ids for i in f})
    gt_map = {v: k for k, v in enumerate(gt_id_set)}
    tr_map = {v: k for k, v in enumerate(tr_id_set)}

    gt_ids, tracker_ids, similarity = [], [], []
    for t in range(seq.num_frames):
        gt_ids.append(np.array([gt_map[i] for i in seq.gt_ids[t]], dtype=int))
        tracker_ids.append(np.array([tr_map[i] for i in seq.pred_ids[t]], dtype=int))
        if sims is None:
            arr = _np_iou_matrix(seq.gt_boxes[t], seq.pred_boxes[t])
        else:
            arr = np.asarray(sims[t], dtype=float).reshape(
                len(seq.gt_boxes[t]), len(seq.pred_boxes[t])
            )
        similarity.append(arr)

    return {
        "num_timesteps": seq.num_frames,
        "num_gt_ids": len(gt_id_set),
        "num_tracker_ids": len(tr_id_set),
        "num_gt_dets": seq.num_gt_dets,
        "num_tracker_dets": seq.num_pred_dets,
        "gt_ids": gt_ids,
        "tracker_ids": tracker_ids,
        "similarity_scores": similarity,
    }


def _trackeval_all(seq: Sequence, sims: list | None = None) -> dict[str, float]:
    data = _trackeval_data(seq, sims=sims)
    clear = TE_CLEAR({"THRESHOLD": IOU_THRESHOLD, "PRINT_CONFIG": False}).eval_sequence(
        data
    )
    ident = TE_Identity(
        {"THRESHOLD": IOU_THRESHOLD, "PRINT_CONFIG": False}
    ).eval_sequence(data)
    hota = TE_HOTA({"PRINT_CONFIG": False}).eval_sequence(data)
    return {
        "MOTA": float(clear["MOTA"]),
        "MOTP": float(clear["MOTP"]),
        "IDSW": float(clear["IDSW"]),
        "IDF1": float(ident["IDF1"]),
        "HOTA": float(np.mean(hota["HOTA"])),
        "DetA": float(np.mean(hota["DetA"])),
        "AssA": float(np.mean(hota["AssA"])),
    }


def _trackeval_clear_identity(seq: Sequence) -> None:
    """Run only CLEAR + Identity (timing helper for the fair 3-way comparison)."""
    data = _trackeval_data(seq)
    TE_CLEAR({"THRESHOLD": IOU_THRESHOLD, "PRINT_CONFIG": False}).eval_sequence(data)
    TE_Identity({"THRESHOLD": IOU_THRESHOLD, "PRINT_CONFIG": False}).eval_sequence(data)


# --- py-motmetrics --------------------------------------------------------------


def _iou_distance(gt_boxes: list, pred_boxes: list):  # -> np.ndarray
    """IoU distance matrix (1 - IoU) with sub-threshold pairs masked to NaN.

    Built directly rather than via ``mm.distances.iou_matrix`` (which is not
    NumPy 2 compatible).
    """
    iou = _np_iou_matrix(gt_boxes, pred_boxes)
    dist = 1.0 - iou
    dist[iou < IOU_THRESHOLD] = np.nan
    return dist


def _motmetrics_all(seq: Sequence) -> dict[str, float]:
    acc = mm.MOTAccumulator(auto_id=True)
    for t in range(seq.num_frames):
        dists = _iou_distance(seq.gt_boxes[t], seq.pred_boxes[t])
        acc.update(seq.gt_ids[t], seq.pred_ids[t], dists)
    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=["mota", "motp", "num_switches", "idf1"],
        name="seq",
    )
    row = summary.iloc[0]
    # motmetrics MOTP is a distance (1 - IoU); report as IoU-based to match others.
    return {
        "MOTA": float(row["mota"]),
        "MOTP": float(1.0 - row["motp"]),
        "IDSW": float(row["num_switches"]),
        "IDF1": float(row["idf1"]),
    }


# --- parity ---------------------------------------------------------------------


# Cross-implementation tolerance for the summary metrics. On dense real data,
# Hungarian assignment ties are resolved differently by each solver, so a couple
# of matches/switches shift and metrics differ by ~1e-4. That is not a bug (the
# unit tests in tests/test_parity.py enforce exact 1e-9 parity on tie-free
# synthetic data); this gate only needs to catch real numeric divergence.
_PARITY_ATOL = 5e-3


def _check_parity(seq: Sequence, results: dict[str, dict[str, float]]) -> list[str]:
    """Flag summary-metric disagreements beyond assignment tie-breaking noise."""
    notes: list[str] = []
    base = results["motrics"]
    for engine, keys in (
        ("trackeval", ("MOTA", "MOTP", "IDF1", "HOTA", "DetA", "AssA")),
        ("motmetrics", ("MOTA", "IDF1")),
    ):
        other = results.get(engine)
        if other is None:
            continue
        for key in keys:
            if (
                key in base
                and key in other
                and abs(base[key] - other[key]) > _PARITY_ATOL
            ):
                notes.append(
                    f"  ⚠ {key}: motrics={base[key]:.6f} {engine}={other[key]:.6f}"
                )
    return notes


# --- driver ---------------------------------------------------------------------


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1e3:8.2f}"


def run(sequences: list[Sequence], repeats: int) -> int:
    if motrics.is_debug_build():
        print(
            "WARNING: debug build — timings are not meaningful (~10x slow). "
            "Rebuild: uv run maturin develop --release --uv"
        )

    engines = ["motrics"]
    if _HAS_TRACKEVAL:
        engines.append("trackeval")
    else:
        print("note: TrackEval not installed — skipping (uv sync --group parity)")
    if _HAS_MOTMETRICS:
        engines.append("motmetrics")
    else:
        print("note: py-motmetrics not installed — skipping (uv sync --group parity)")

    print(f"\n{len(sequences)} sequence(s), {repeats} repeat(s)\n")

    # Accumulate totals for a closing summary.
    totals: dict[str, float] = dict.fromkeys(engines, 0.0)
    ci_totals: dict[str, float] = dict.fromkeys(engines, 0.0)  # CLEAR+Identity only
    parity_ok = True

    for seq in sequences:
        print(
            f"{seq.name}  ({seq.num_frames} frames, "
            f"{seq.num_gt_dets} gt / {seq.num_pred_dets} pred dets)"
        )

        results: dict[str, dict[str, float]] = {}
        times: dict[str, float] = {}
        ci_times: dict[str, float] = {}

        # motrics: three separate calls (s=seq binds the loop var in each lambda).
        (c, t_c) = _time(lambda s=seq: _motrics_clear(s), repeats)
        (i, t_i) = _time(lambda s=seq: _motrics_identity(s), repeats)
        (h, t_h) = _time(lambda s=seq: _motrics_hota(s), repeats)
        results["motrics"] = {**c, **i, **h}
        times["motrics"] = t_c + t_i + t_h
        ci_times["motrics"] = t_c + t_i

        if _HAS_TRACKEVAL:
            # Parity: feed TrackEval identical similarities (motrics' own IoU) so
            # any discrepancy is pure metric math. Timing: let TrackEval compute
            # its own IoU for a fair end-to-end comparison.
            shared_sims = [
                motrics.iou_matrix(seq.gt_boxes[t], seq.pred_boxes[t])
                for t in range(seq.num_frames)
            ]
            results["trackeval"] = _trackeval_all(seq, sims=shared_sims)
            (_, t) = _time(lambda s=seq: _trackeval_all(s), repeats)
            times["trackeval"] = t
            # CLEAR+Identity time: build the data dict (incl. IoU) then run just
            # those two metrics, for a fair 3-way comparison against motmetrics.
            (_, t_ci) = _time(lambda s=seq: _trackeval_clear_identity(s), repeats)
            ci_times["trackeval"] = t_ci

        if _HAS_MOTMETRICS:
            (r, t) = _time(lambda s=seq: _motmetrics_all(s), repeats)
            results["motmetrics"] = r
            times["motmetrics"] = t
            ci_times["motmetrics"] = t  # motmetrics computes only CLEAR+Identity

        # Report metric values (motrics as the reference row).
        ref = results["motrics"]
        print(
            f"    metrics: MOTA={ref['MOTA']:.4f} MOTP={ref['MOTP']:.4f} "
            f"IDF1={ref['IDF1']:.4f} HOTA={ref['HOTA']:.4f}"
        )

        # Parity.
        notes = _check_parity(seq, results)
        if notes:
            parity_ok = False
            print("    parity: MISMATCH")
            for line in notes:
                print("  " + line)
        else:
            print("    parity: OK (all engines agree within tolerance)")

        # Timing table (* speedup = CLEAR+Identity time relative to motrics).
        print(f"    {'engine':<12}{'all families':>14}{'CLEAR+Id':>12}{'speedup*':>10}")
        for e in engines:
            covers = "no HOTA" if e == "motmetrics" else ""
            all_ms = _fmt_ms(times[e])
            ci_ms = _fmt_ms(ci_times.get(e, times[e]))
            speed = (
                ci_times.get(e, 0.0) / ci_times["motrics"] if ci_times.get(e) else 1.0
            )
            print(f"    {e:<12}{all_ms:>12}ms{ci_ms:>10}ms{speed:>9.1f}x  {covers}")
            totals[e] += times[e]
            ci_totals[e] += ci_times.get(e, 0.0)
        print()

    # Summary.
    print("=" * 60)
    print("SUMMARY (speedup = engine time / motrics time; higher = motrics faster)")
    print(f"  {'engine':<12}{'CLEAR+Identity':>18}{'speedup':>10}")
    for e in engines:
        if ci_totals[e] <= 0:
            continue
        speed = ci_totals[e] / ci_totals["motrics"]
        print(f"  {e:<12}{_fmt_ms(ci_totals[e]):>16}ms{speed:>9.1f}x")
    if _HAS_TRACKEVAL:
        full = totals["trackeval"] / totals["motrics"]
        print(
            f"\n  Full pipeline incl. HOTA (motrics vs TrackEval): {full:.1f}x faster"
        )
    print("=" * 60)

    if not parity_ok:
        print("\nPARITY FAILURES DETECTED — see mismatches above.")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeats", type=int, default=5, help="timing repeats (best time is reported)"
    )
    parser.add_argument(
        "--smoke", action="store_true", help="single repeat, just verify it runs"
    )
    args = parser.parse_args()

    if np is None:
        print("error: numpy is required (uv sync --group parity)")
        return 1

    sequences = load_real()
    if not sequences:
        print("error: no sequences found. Fetch them first:")
        print("  uv run python benchmarks/download.py")
        return 1
    return run(sequences, 1 if args.smoke else args.repeats)


if __name__ == "__main__":
    raise SystemExit(main())
