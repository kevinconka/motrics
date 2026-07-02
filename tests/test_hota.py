"""Tests for HOTA metrics."""

import motrics
import pytest

A = (0.0, 0.0, 10.0, 10.0)


def test_perfect_tracking_scores_one() -> None:
    ids = [[1], [1]]
    boxes = [[A], [A]]
    m = motrics.compute_hota(ids, boxes, ids, boxes)
    assert m.hota == pytest.approx(1.0)
    assert m.deta == pytest.approx(1.0)
    assert m.assa == pytest.approx(1.0)
    assert m.loca == pytest.approx(1.0)
    assert len(m.alphas) == 19
    assert len(m.hota_alphas) == 19


def test_id_split_keeps_deta_but_halves_assa() -> None:
    # One perfectly localised gt object covered by two predicted ids:
    # detection is perfect, association halves.
    gt_ids = [[1], [1], [1], [1]]
    gt_boxes = [[A], [A], [A], [A]]
    pred_ids = [[10], [10], [20], [20]]
    pred_boxes = [[A], [A], [A], [A]]
    m = motrics.compute_hota(gt_ids, gt_boxes, pred_ids, pred_boxes)
    assert m.deta == pytest.approx(1.0)
    assert m.assa == pytest.approx(0.5)
    assert m.hota == pytest.approx(0.5**0.5)
    assert m.loca == pytest.approx(1.0)


def test_all_false_positives() -> None:
    m = motrics.compute_hota([[]], [[]], [[9]], [[A]])
    assert m.hota == 0.0
    assert m.num_pred == 1


def test_all_misses() -> None:
    m = motrics.compute_hota([[1]], [[A]], [[]], [[]])
    assert m.hota == 0.0
    assert m.num_gt == 1


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same number of frames"):
        motrics.compute_hota([[1], [1]], [[A], [A]], [[1]], [[A]])


def test_repr() -> None:
    m = motrics.compute_hota([[1]], [[A]], [[1]], [[A]])
    assert "HotaMetrics(" in repr(m)
