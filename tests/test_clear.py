"""Tests for CLEAR MOT metrics."""

import motrics
import pytest

A = (0.0, 0.0, 10.0, 10.0)
B = (20.0, 20.0, 30.0, 30.0)


def test_perfect_tracking() -> None:
    gt_ids = [[1, 2], [1, 2]]
    gt_boxes = [[A, B], [A, B]]
    m = motrics.compute_clear(gt_ids, gt_boxes, gt_ids, gt_boxes)
    assert m.num_matches == 4
    assert m.num_false_positives == 0
    assert m.num_misses == 0
    assert m.num_switches == 0
    assert m.mota == pytest.approx(1.0)
    assert m.motp == pytest.approx(1.0)
    assert m.num_frames == 2
    assert m.num_gt == 4


def test_all_missed() -> None:
    m = motrics.compute_clear([[1]], [[A]], [[]], [[]])
    assert m.num_misses == 1
    assert m.num_matches == 0
    assert m.mota == pytest.approx(0.0)  # 1 - 1/1


def test_false_positive() -> None:
    m = motrics.compute_clear([[]], [[]], [[9]], [[A]])
    assert m.num_false_positives == 1
    assert m.num_gt == 0


def test_identity_switch() -> None:
    # gt object 1 is matched to hypothesis 10, then 20.
    m = motrics.compute_clear([[1], [1]], [[A], [A]], [[10], [20]], [[A], [A]])
    assert m.num_matches == 2
    assert m.num_switches == 1
    assert m.mota == pytest.approx(0.5)  # 1 - 1/2


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same number of frames"):
        motrics.compute_clear([[1], [1]], [[A], [A]], [[1]], [[A]])


def test_frame_id_box_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        motrics.compute_clear([[1, 2]], [[A]], [[1]], [[A]])


def test_repr() -> None:
    m = motrics.compute_clear([[1]], [[A]], [[1]], [[A]])
    assert "ClearMetrics(" in repr(m)
