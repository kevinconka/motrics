"""Tests for `Frames`, `evaluate()`, and `compute_hota_from_similarity`.

`evaluate()` is meant to be a pure convenience/perf wrapper — same numbers as
calling `compute_clear`/`compute_identity`/`compute_hota` separately, just
sharing one similarity matrix instead of building it three times. These tests
check exactly that: bit-exact agreement, not new metric behaviour.
"""

from __future__ import annotations

import motrics
import pytest

from benchmarks.fixtures import Sequence, make_synthetic

SEQUENCES = make_synthetic()


def test_frames_num_frames_and_num_dets() -> None:
    seq = SEQUENCES[0]
    gt = motrics.Frames(ids=seq.gt_ids, boxes=seq.gt_boxes)
    assert gt.num_frames == seq.num_frames
    assert gt.num_dets == seq.num_gt_dets


def test_frames_mismatched_lengths_raises() -> None:
    with pytest.raises(ValueError, match="same number of frames"):
        motrics.Frames(ids=[[1]], boxes=[[(0.0, 0.0, 1.0, 1.0)], []])


@pytest.mark.parametrize("seq", SEQUENCES, ids=lambda s: s.name)
def test_evaluate_matches_separate_calls(seq: Sequence) -> None:
    gt = motrics.Frames(ids=seq.gt_ids, boxes=seq.gt_boxes)
    pred = motrics.Frames(ids=seq.pred_ids, boxes=seq.pred_boxes)
    result = motrics.evaluate(gt, pred)

    clear = motrics.compute_clear(
        seq.gt_ids, seq.gt_boxes, seq.pred_ids, seq.pred_boxes
    )
    identity = motrics.compute_identity(
        seq.gt_ids, seq.gt_boxes, seq.pred_ids, seq.pred_boxes
    )
    hota = motrics.compute_hota(seq.gt_ids, seq.gt_boxes, seq.pred_ids, seq.pred_boxes)

    assert result.clear.mota == clear.mota
    assert result.clear.motp == clear.motp
    assert result.clear.num_switches == clear.num_switches
    assert result.identity.idf1 == identity.idf1
    assert result.hota.hota == hota.hota
    assert result.hota.deta == hota.deta
    assert result.hota.assa == hota.assa


def test_evaluate_respects_iou_threshold() -> None:
    seq = SEQUENCES[0]
    gt = motrics.Frames(ids=seq.gt_ids, boxes=seq.gt_boxes)
    pred = motrics.Frames(ids=seq.pred_ids, boxes=seq.pred_boxes)

    loose = motrics.evaluate(gt, pred, iou_threshold=0.1)
    strict = motrics.evaluate(gt, pred, iou_threshold=0.99)
    assert loose.clear.mota != strict.clear.mota
    assert loose.identity.idf1 != strict.identity.idf1


@pytest.mark.parametrize("seq", SEQUENCES, ids=lambda s: s.name)
def test_hota_from_similarity_matches_from_boxes(seq: Sequence) -> None:
    similarity = [
        motrics.iou_matrix(g, p)
        for g, p in zip(seq.gt_boxes, seq.pred_boxes, strict=True)
    ]
    from_similarity = motrics.compute_hota_from_similarity(
        seq.gt_ids, seq.pred_ids, similarity
    )
    from_boxes = motrics.compute_hota(
        seq.gt_ids, seq.gt_boxes, seq.pred_ids, seq.pred_boxes
    )

    assert from_similarity.hota == pytest.approx(from_boxes.hota)
    assert from_similarity.deta == pytest.approx(from_boxes.deta)
    assert from_similarity.assa == pytest.approx(from_boxes.assa)
    assert from_similarity.num_gt == from_boxes.num_gt
    assert from_similarity.num_pred == from_boxes.num_pred
