"""Tests for the streaming `Accumulator`.

The accumulator is meant to produce exactly the same CLEAR and Identity
numbers as the batch path — feeding frames one at a time must equal computing
over the whole sequence at once. These tests check that bit-exact agreement
(against `evaluate()` and the `compute_*` functions), plus the incremental-API
behaviour and input validation.
"""

from __future__ import annotations

import motrics
import numpy as np
import pytest

from benchmarks.fixtures import Sequence, make_synthetic

SEQUENCES = make_synthetic()


def _feed(seq: Sequence) -> motrics.AccumulatorResult:
    acc = motrics.Accumulator()
    for g_ids, g_boxes, p_ids, p_boxes in zip(
        seq.gt_ids, seq.gt_boxes, seq.pred_ids, seq.pred_boxes, strict=True
    ):
        acc.update(g_ids, g_boxes, p_ids, p_boxes)
    return acc.compute()


@pytest.mark.parametrize("seq", SEQUENCES, ids=lambda s: s.name)
def test_streaming_matches_batch(seq: Sequence) -> None:
    result = _feed(seq)

    clear = motrics.compute_clear(
        seq.gt_ids, seq.gt_boxes, seq.pred_ids, seq.pred_boxes
    )
    identity = motrics.compute_identity(
        seq.gt_ids, seq.gt_boxes, seq.pred_ids, seq.pred_boxes
    )

    assert result.clear.mota == clear.mota
    assert result.clear.motp == clear.motp
    assert result.clear.num_matches == clear.num_matches
    assert result.clear.num_false_positives == clear.num_false_positives
    assert result.clear.num_misses == clear.num_misses
    assert result.clear.num_switches == clear.num_switches
    assert result.clear.num_frames == clear.num_frames
    assert result.identity.idf1 == identity.idf1
    assert result.identity.idtp == identity.idtp
    assert result.identity.idfp == identity.idfp
    assert result.identity.idfn == identity.idfn


@pytest.mark.parametrize("seq", SEQUENCES, ids=lambda s: s.name)
def test_update_from_similarity_matches_update(seq: Sequence) -> None:
    boxes_result = _feed(seq)

    acc = motrics.Accumulator()
    for g_ids, g_boxes, p_ids, p_boxes in zip(
        seq.gt_ids, seq.gt_boxes, seq.pred_ids, seq.pred_boxes, strict=True
    ):
        acc.update_from_similarity(g_ids, p_ids, motrics.iou_matrix(g_boxes, p_boxes))
    sim_result = acc.compute()

    assert sim_result.clear.mota == boxes_result.clear.mota
    assert sim_result.clear.num_switches == boxes_result.clear.num_switches
    assert sim_result.identity.idf1 == boxes_result.identity.idf1


def test_numpy_boxes_match_lists() -> None:
    seq = SEQUENCES[0]
    acc = motrics.Accumulator()
    for g_ids, g_boxes, p_ids, p_boxes in zip(
        seq.gt_ids, seq.gt_boxes, seq.pred_ids, seq.pred_boxes, strict=True
    ):
        gt = np.asarray(g_boxes, dtype=np.float64).reshape(-1, 4)
        pred = np.asarray(p_boxes, dtype=np.float64).reshape(-1, 4)
        acc.update(g_ids, gt, p_ids, pred)
    result = acc.compute()

    expected = _feed(seq)
    assert result.clear.mota == expected.clear.mota
    assert result.identity.idf1 == expected.identity.idf1


def test_xywh_matches_xyxy() -> None:
    gt_ids, pred_ids = [1], [1]
    xyxy = [(0.0, 0.0, 10.0, 10.0)]
    xywh = [(0.0, 0.0, 10.0, 10.0)]

    a = motrics.Accumulator(box_format="xyxy")
    a.update(gt_ids, xyxy, pred_ids, xyxy)
    b = motrics.Accumulator(box_format="xywh")
    b.update(gt_ids, xywh, pred_ids, xywh)

    assert a.compute().clear.mota == b.compute().clear.mota == 1.0


def test_num_frames_and_repeatable_compute() -> None:
    acc = motrics.Accumulator()
    assert acc.num_frames == 0

    acc.update([1], [(0.0, 0.0, 10.0, 10.0)], [1], [(0.0, 0.0, 10.0, 10.0)])
    assert acc.num_frames == 1
    first = acc.compute()

    # compute() does not consume state; a second identical frame changes it.
    acc.update([1], [(0.0, 0.0, 10.0, 10.0)], [1], [(0.0, 0.0, 10.0, 10.0)])
    assert acc.num_frames == 2
    second = acc.compute()

    assert first.clear.num_matches == 1
    assert second.clear.num_matches == 2


def test_empty_accumulator() -> None:
    result = motrics.Accumulator().compute()
    assert result.clear.num_frames == 0
    assert result.clear.mota == 0.0
    assert result.identity.idf1 == 0.0


def test_gt_length_mismatch_raises() -> None:
    acc = motrics.Accumulator()
    with pytest.raises(ValueError, match="gt_ids and gt_boxes length mismatch"):
        acc.update([1, 2], [(0.0, 0.0, 1.0, 1.0)], [], [])


def test_pred_length_mismatch_raises() -> None:
    acc = motrics.Accumulator()
    with pytest.raises(ValueError, match="pred_ids and pred_boxes length mismatch"):
        acc.update([], [], [1], [])


def test_similarity_shape_mismatch_raises() -> None:
    acc = motrics.Accumulator()
    with pytest.raises(ValueError, match="similarity has 1 rows, expected 2"):
        acc.update_from_similarity([1, 2], [3], [[0.9]])
    with pytest.raises(ValueError, match="similarity row 0 has 1 columns, expected 2"):
        acc.update_from_similarity([1], [3, 4], [[0.9]])


def test_invalid_box_format_raises() -> None:
    with pytest.raises(ValueError, match="unknown box_format"):
        motrics.Accumulator(box_format="polar")  # ty: ignore[invalid-argument-type]


def test_repr() -> None:
    acc = motrics.Accumulator()
    acc.update([1], [(0.0, 0.0, 10.0, 10.0)], [1], [(0.0, 0.0, 10.0, 10.0)])
    assert repr(acc) == "Accumulator(num_frames=1)"

    result = acc.compute()
    text = repr(result)
    assert text.startswith("AccumulatorResult(")
    assert "clear=" in text
    assert "identity=" in text
