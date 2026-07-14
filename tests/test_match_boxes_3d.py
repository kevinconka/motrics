"""Tests for the 3D-box-matching primitive (`match_boxes_3d`)."""

import math

import motrics
import numpy as np
import pytest


def _box(x: float) -> tuple:
    return (x, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0)


def test_perfect_matches() -> None:
    boxes = [_box(0.0), _box(10.0)]
    result = motrics.match_boxes_3d(boxes, boxes)
    assert result.matches == [(0, 0), (1, 1)]
    assert result.scores == pytest.approx([1.0, 1.0])
    assert result.unmatched_a == []
    assert result.unmatched_b == []


def test_disjoint_boxes_stay_unmatched() -> None:
    result = motrics.match_boxes_3d([_box(0.0)], [_box(100.0)])
    assert result.matches == []
    assert result.unmatched_a == [0]
    assert result.unmatched_b == [0]


def test_unequal_set_sizes() -> None:
    a = [_box(0.0), _box(10.0)]
    b = [_box(0.0)]
    result = motrics.match_boxes_3d(a, b)
    assert result.matches == [(0, 0)]
    assert result.unmatched_a == [1]
    assert result.unmatched_b == []


def test_greedy_method() -> None:
    result = motrics.match_boxes_3d([_box(0.0)], [_box(0.0)], method="greedy")
    assert result.matches == [(0, 0)]


def test_yaw_reduces_overlap_of_rotated_square_footprint() -> None:
    # Same square footprint (l == w) rotated 45 degrees at the same centre:
    # the classic intersection-over-union of a square and its own diagonal
    # rotation is sqrt(2)/2, so this checks yaw is actually used, not ignored.
    a = (0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0)
    b = (0.0, 0.0, 0.0, 2.0, 2.0, 2.0, math.pi / 4)
    result = motrics.match_boxes_3d([a], [b])
    assert result.matches == [(0, 0)]
    assert result.scores == pytest.approx([math.sqrt(2) / 2])


def test_hungarian_beats_greedy_on_conflict() -> None:
    # Mirrors assignment.rs's hungarian_beats_greedy_on_conflict: greedy grabs
    # the single best pair (g0, p0) first, stranding g1 with a below-threshold
    # leftover; Hungarian instead finds the higher-scoring anti-diagonal.
    g0, p0 = _box(0.0), _box(0.2)  # iou 9/11 ~ 0.818
    p1, g1 = _box(-0.6), _box(0.7)  # iou(g0, p1) 7/13 ~ 0.538, iou(g1, p0) 0.6

    greedy = motrics.match_boxes_3d([g0, g1], [p0, p1], method="greedy")
    assert greedy.matches == [(0, 0)]
    assert greedy.scores == pytest.approx([9 / 11])

    hungarian = motrics.match_boxes_3d([g0, g1], [p0, p1], method="hungarian")
    assert hungarian.matches == [(0, 1), (1, 0)]
    assert hungarian.scores == pytest.approx([7 / 13, 3 / 5])


def test_empty_inputs() -> None:
    result = motrics.match_boxes_3d([], [_box(0.0)])
    assert result.matches == []
    assert result.unmatched_a == []
    assert result.unmatched_b == [0]


def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="unknown method"):
        motrics.match_boxes_3d([], [], method="bogus")


def test_repr() -> None:
    boxes = [_box(0.0)]
    assert "Matching(" in repr(motrics.match_boxes_3d(boxes, boxes))


def test_numpy_boxes_match_list_of_tuples() -> None:
    boxes = [_box(0.0), _box(10.0)]
    from_list = motrics.match_boxes_3d(boxes, boxes)
    from_numpy = motrics.match_boxes_3d(
        np.asarray(boxes, dtype=np.float64), np.asarray(boxes, dtype=np.float64)
    )
    assert from_numpy.matches == from_list.matches
    assert from_numpy.scores == pytest.approx(from_list.scores)


def test_numpy_wrong_shape_raises() -> None:
    bad = np.zeros((2, 4), dtype=np.float64)
    with pytest.raises(ValueError, match=r"shape \(N, 7\)"):
        motrics.match_boxes_3d(bad, bad)


def test_numpy_wrong_shape_in_second_argument_raises() -> None:
    # boxes_a is valid, so this exercises boxes_b's own shape check rather
    # than short-circuiting on boxes_a's.
    good = np.asarray([_box(0.0)], dtype=np.float64)
    bad = np.zeros((1, 4), dtype=np.float64)
    with pytest.raises(ValueError, match=r"shape \(N, 7\)"):
        motrics.match_boxes_3d(good, bad)
