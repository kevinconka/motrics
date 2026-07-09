"""Numeric parity for ``motrics.compat.motmetrics`` against real py-motmetrics.

Both accumulators are fed the same distance matrix, built from
``motrics.iou_matrix`` rather than ``motmetrics.distances.iou_matrix`` (whose
own ``np.asfarray`` call is broken on NumPy 2.0), so any difference is a
genuine algorithmic discrepancy, not an IoU-implementation difference.
"""

from __future__ import annotations

import math
from typing import Any

import motrics
import motrics.compat.motmetrics as compat_mm
import pytest

from benchmarks.fixtures import Sequence, make_synthetic

motmetrics = pytest.importorskip("motmetrics")
pytest.importorskip("pandas")

IOU_THRESHOLD = 0.5


def _dist_matrix(gt_boxes: list, pred_boxes: list) -> list[list[float]]:
    if not gt_boxes or not pred_boxes:
        return []
    sims = motrics.iou_matrix(gt_boxes, pred_boxes)
    return [
        [(1.0 - s) if s >= IOU_THRESHOLD else math.nan for s in row] for row in sims
    ]


def _accumulate(seq: Sequence, acc: Any) -> None:
    for t in range(seq.num_frames):
        dists = _dist_matrix(seq.gt_boxes[t], seq.pred_boxes[t])
        acc.update(seq.gt_ids[t], seq.pred_ids[t], dists, frameid=t)


def _same(got: float, ref: float) -> bool:
    if got != got and ref != ref:  # both NaN
        return True
    return got == pytest.approx(ref, abs=1e-9)


@pytest.mark.parametrize("seq", make_synthetic(), ids=lambda s: s.name)
def test_motmetrics_compat_parity(seq: Sequence) -> None:
    ref_acc = motmetrics.MOTAccumulator()
    got_acc = compat_mm.MOTAccumulator()
    _accumulate(seq, ref_acc)
    _accumulate(seq, got_acc)

    metrics = list(compat_mm.metrics.SUPPORTED)
    ref = motmetrics.metrics.create().compute(ref_acc, metrics=metrics, name="ref")
    got = compat_mm.metrics.create().compute(got_acc, metrics=metrics, name="got")

    for name in metrics:
        assert _same(got[name].iloc[0], ref[name].iloc[0]), (
            f"{name}: got={got[name].iloc[0]!r} ref={ref[name].iloc[0]!r}"
        )


def test_metrics_compute_rejects_unsupported_metric() -> None:
    acc = compat_mm.MOTAccumulator(auto_id=True)
    acc.update([], [], [])
    with pytest.raises(NotImplementedError):
        compat_mm.metrics.create().compute(acc, metrics=["num_transfer"])


def test_continuity_survives_large_magnitude_distances() -> None:
    """A fixed continuity bonus would lose here: raw-score gap (1500) > 1000."""
    acc = compat_mm.MOTAccumulator(auto_id=True)
    acc.update([0], [10], [[0.0]])
    acc.update([0], [10, 20], [[2000.0, 500.0]])

    summary = compat_mm.metrics.create().compute(acc, metrics=["num_switches"])
    assert summary["num_switches"].iloc[0] == 0


def test_accumulator_auto_id_and_manual_id_are_mutually_exclusive() -> None:
    auto = compat_mm.MOTAccumulator(auto_id=True)
    with pytest.raises(AssertionError):
        auto.update([], [], [], frameid=0)

    manual = compat_mm.MOTAccumulator(auto_id=False)
    with pytest.raises(AssertionError):
        manual.update([], [], [])
