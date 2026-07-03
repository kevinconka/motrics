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
from pathlib import Path
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


def _trackeval_data(seq: Sequence) -> dict[str, Any]:
    """Build TrackEval's raw-data dict once per sequence (id remapping +
    similarity matrix) — shared prep, not attributed to any one metric, exactly
    like TrackEval's own `Evaluator` builds it once then loops over metrics."""
    boxes = zip(seq.gt_boxes, seq.pred_boxes, strict=True)
    return {
        "num_timesteps": seq.num_frames,
        "num_gt_ids": len({i for f in seq.gt_ids for i in f}),
        "num_tracker_ids": len({i for f in seq.pred_ids for i in f}),
        "num_gt_dets": seq.num_gt_dets,
        "num_tracker_dets": seq.num_pred_dets,
        "gt_ids": _contiguous(seq.gt_ids),
        "tracker_ids": _contiguous(seq.pred_ids),
        "similarity_scores": [_np_iou(g, p) for g, p in boxes],
    }


def _trackeval_clear(data: dict[str, Any]) -> dict[str, float]:
    cfg = {"THRESHOLD": IOU_THRESHOLD, "PRINT_CONFIG": False}
    r = CLEAR(cfg).eval_sequence(data)
    return {"MOTA": float(r["MOTA"]), "MOTP": float(r["MOTP"])}


def _trackeval_identity(data: dict[str, Any]) -> dict[str, float]:
    cfg = {"THRESHOLD": IOU_THRESHOLD, "PRINT_CONFIG": False}
    return {"IDF1": float(Identity(cfg).eval_sequence(data)["IDF1"])}


def _trackeval_hota(data: dict[str, Any]) -> dict[str, float]:
    r = HOTA({"PRINT_CONFIG": False}).eval_sequence(data)
    return {
        "HOTA": float(np.mean(r["HOTA"])),
        "DetA": float(np.mean(r["DetA"])),
        "AssA": float(np.mean(r["AssA"])),
    }


def _motmetrics_acc(seq: Sequence) -> Any:
    """Build the accumulator once per sequence — shared prep (each `update()`
    call already does its own assignment), not attributed to any one metric."""
    acc = mm.MOTAccumulator(auto_id=True)
    frames = zip(seq.gt_boxes, seq.pred_boxes, seq.gt_ids, seq.pred_ids, strict=True)
    for g, p, gi, pi in frames:
        iou = _np_iou(g, p)
        # motmetrics wants 1 - IoU, with sub-threshold pairs masked out.
        acc.update(gi, pi, np.where(iou >= IOU_THRESHOLD, 1.0 - iou, np.nan))
    return acc


def _motmetrics_mota(acc: Any) -> dict[str, float]:
    row = mm.metrics.create().compute(acc, metrics=["mota"], name="seq").iloc[0]
    return {"MOTA": float(row["mota"])}


def _motmetrics_idf1(acc: Any) -> dict[str, float]:
    row = mm.metrics.create().compute(acc, metrics=["idf1"], name="seq").iloc[0]
    return {"IDF1": float(row["idf1"])}


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


def _fmt_ms(seconds: float | None) -> str:
    return f"{seconds * 1000:.0f} ms" if seconds is not None else "—"


def _fmt_speedup(other: float | None, motrics_time: float) -> str:
    return f"{other / motrics_time:.1f}x" if other is not None else "—"


def _speedup_rows(times: dict[str, dict[str, float]]) -> list[tuple[str, ...]]:
    """Per-metric total time (across every sequence and repeat) for each
    engine, excluding shared prep that isn't attributed to any one metric
    (TrackEval's similarity matrix, motmetrics' accumulator build) — the same
    "pre-aligned arrays" methodology already documented in this directory's
    README, just broken out per metric instead of bundled."""
    rows = []
    for metric in ("CLEAR", "Identity", "HOTA"):
        m = times["motrics"][metric]
        te = times["trackeval"].get(metric)
        mmr = times["motmetrics"].get(metric)
        rows.append(
            (
                metric,
                _fmt_ms(m),
                _fmt_ms(te),
                _fmt_ms(mmr),
                _fmt_speedup(te, m),
                _fmt_speedup(mmr, m),
            )
        )
    return rows


def _print_speedup_table(times: dict[str, dict[str, float]]) -> None:
    header = ("Metric", "motrics", "TrackEval", "py-motmetrics", "vs TE", "vs mm")
    rows = [header, *_speedup_rows(times)]
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    for row in rows:
        cells = [
            row[0].ljust(widths[0]),
            *(c.rjust(w) for c, w in zip(row[1:], widths[1:], strict=True)),
        ]
        print("  ".join(cells))


def _markdown_speedup_table(times: dict[str, dict[str, float]]) -> list[str]:
    columns = [
        "Metric",
        "motrics",
        "TrackEval",
        "py-motmetrics",
        "vs TrackEval",
        "vs py-motmetrics",
    ]
    lines = [
        f"| {' | '.join(columns)} |",
        f"| {' | '.join(['---'] + [':---:'] * (len(columns) - 1))} |",
    ]
    lines += [f"| {' | '.join(row)} |" for row in _speedup_rows(times)]
    return lines


def _write_markdown(
    path: Path, rows: list[dict[str, Any]], times: dict[str, dict[str, float]]
) -> None:
    """Render the same numbers `run()` prints as a markdown report (for the CI
    sticky-comment step); purely a formatting concern, no new computation."""
    lines = ["### motrics benchmark — MOT17 data", ""]
    if motrics.is_debug_build():
        lines += ["> ⚠️ debug build — timings are not meaningful (~10x slower).", ""]
    lines += [
        *_markdown_speedup_table(times),
        "",
        "<details>",
        "<summary>Per-sequence results</summary>",
        "",
        "| Sequence | Frames | GT dets | Pred dets | MOTA | IDF1 | HOTA | Parity |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for r in rows:
        parity = "✅" if r["ok"] else "⚠️"
        lines.append(
            f"| {r['name']} | {r['frames']} | {r['gt_dets']} | {r['pred_dets']} | "
            f"{r['mota']:.3f} | {r['idf1']:.3f} | {r['hota']:.3f} | {parity} |"
        )
    lines += ["", "</details>", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(sequences: list[Sequence], repeats: int, markdown: Path | None = None) -> int:
    if motrics.is_debug_build():
        print(
            "WARNING: debug build — timings are not meaningful (~10x slow). "
            "Rebuild: uv run maturin develop --release --uv"
        )
    print(f"\n{len(sequences)} sequence(s), {repeats} repeat(s)\n")

    times: dict[str, dict[str, float]] = {
        "motrics": {"CLEAR": 0.0, "Identity": 0.0, "HOTA": 0.0},
        "trackeval": {"CLEAR": 0.0, "Identity": 0.0, "HOTA": 0.0},
        "motmetrics": {"CLEAR": 0.0, "Identity": 0.0},
    }
    ok = True
    rows: list[dict[str, Any]] = []
    for seq in sequences:
        c, t_mc = _time(lambda s=seq: _motrics_clear(s), repeats)
        i, t_mi = _time(lambda s=seq: _motrics_identity(s), repeats)
        h, t_mh = _time(lambda s=seq: _motrics_hota(s), repeats)

        # Shared prep, built once and excluded from the per-metric timings
        # below — same "pre-aligned arrays" convention as the old bundled
        # numbers (see benchmarks/README.md), just no longer hidden inside a
        # single combined call.
        te_data = _trackeval_data(seq)
        tc, t_tc = _time(lambda d=te_data: _trackeval_clear(d), repeats)
        ti, t_ti = _time(lambda d=te_data: _trackeval_identity(d), repeats)
        th, t_th = _time(lambda d=te_data: _trackeval_hota(d), repeats)

        acc = _motmetrics_acc(seq)
        mmc, t_mmc = _time(lambda a=acc: _motmetrics_mota(a), repeats)
        mmi, t_mmi = _time(lambda a=acc: _motmetrics_idf1(a), repeats)

        m = {**c, **i, **h}
        te = {**tc, **ti, **th}
        mmr = {**mmc, **mmi}
        notes = _check_parity({"motrics": m, "trackeval": te, "motmetrics": mmr})
        seq_ok = not notes
        ok = ok and seq_ok
        print(
            f"{seq.name}  {seq.num_frames} frames, "
            f"{seq.num_gt_dets} gt / {seq.num_pred_dets} pred  |  "
            f"MOTA={m['MOTA']:.3f} IDF1={m['IDF1']:.3f} HOTA={m['HOTA']:.3f}  |  "
            f"parity {'OK' if seq_ok else 'MISMATCH'}"
        )
        for note in notes:
            print(note)
        rows.append(
            {
                "name": seq.name,
                "frames": seq.num_frames,
                "gt_dets": seq.num_gt_dets,
                "pred_dets": seq.num_pred_dets,
                "mota": m["MOTA"],
                "idf1": m["IDF1"],
                "hota": m["HOTA"],
                "ok": seq_ok,
            }
        )

        times["motrics"]["CLEAR"] += t_mc
        times["motrics"]["Identity"] += t_mi
        times["motrics"]["HOTA"] += t_mh
        times["trackeval"]["CLEAR"] += t_tc
        times["trackeval"]["Identity"] += t_ti
        times["trackeval"]["HOTA"] += t_th
        times["motmetrics"]["CLEAR"] += t_mmc
        times["motmetrics"]["Identity"] += t_mmi

    print("\n" + "=" * 60)
    print(f"Per-metric timing over {len(sequences)} sequence(s) (higher = faster):")
    _print_speedup_table(times)
    print("=" * 60)

    if markdown is not None:
        _write_markdown(markdown, rows, times)

    if not ok:
        print("\nPARITY FAILURES DETECTED — see above.")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5, help="timing repeats")
    parser.add_argument("--smoke", action="store_true", help="single repeat")
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="also write a markdown report to this path",
    )
    args = parser.parse_args()

    sequences = load_real()
    if not sequences:
        print("error: no sequences found. Run: uv run python benchmarks/download.py")
        return 1
    return run(sequences, 1 if args.smoke else args.repeats, args.markdown)


if __name__ == "__main__":
    raise SystemExit(main())
