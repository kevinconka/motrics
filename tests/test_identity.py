"""Tests for Identity metrics (IDF1/IDP/IDR)."""

import motrics
import pytest

A = (0.0, 0.0, 10.0, 10.0)


def test_perfect_identity() -> None:
    ids = [[1], [1]]
    boxes = [[A], [A]]
    m = motrics.compute_identity(ids, boxes, ids, boxes)
    assert m.idtp == 2
    assert m.idfp == 0
    assert m.idfn == 0
    assert m.idf1 == pytest.approx(1.0)
    assert m.idp == pytest.approx(1.0)
    assert m.idr == pytest.approx(1.0)


def test_id_split_halves_idf1() -> None:
    # One gt object covered by two different predicted ids -> only half can be
    # globally attributed, so IDF1 drops to 0.5.
    gt_ids = [[1], [1], [1], [1]]
    gt_boxes = [[A], [A], [A], [A]]
    pred_ids = [[10], [10], [20], [20]]
    pred_boxes = [[A], [A], [A], [A]]
    m = motrics.compute_identity(gt_ids, gt_boxes, pred_ids, pred_boxes)
    assert m.idtp == 2
    assert m.idfp == 2
    assert m.idfn == 2
    assert m.idf1 == pytest.approx(0.5)


def test_identity_ignores_per_frame_switches() -> None:
    # CLEAR sees an id switch here, but Identity rewards the globally consistent
    # mapping gt1->10, gt2->20 (no penalty).
    gt_ids = [[1, 2], [1, 2]]
    gt_boxes = [[A, (20.0, 20.0, 30.0, 30.0)]] * 2
    pred_ids = [[10, 20], [10, 20]]
    pred_boxes = gt_boxes
    m = motrics.compute_identity(gt_ids, gt_boxes, pred_ids, pred_boxes)
    assert m.idf1 == pytest.approx(1.0)


def test_all_false_positives() -> None:
    m = motrics.compute_identity([[]], [[]], [[9]], [[A]])
    assert m.idtp == 0
    assert m.idfp == 1
    assert m.idf1 == pytest.approx(0.0)


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same number of frames"):
        motrics.compute_identity([[1], [1]], [[A], [A]], [[1]], [[A]])


def test_repr() -> None:
    m = motrics.compute_identity([[1]], [[A]], [[1]], [[A]])
    assert "IdentityMetrics(" in repr(m)
