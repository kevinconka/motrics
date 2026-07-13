"""Tests for the 3D-box-matching primitive (`match_boxes_3d`)."""

import motrics
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
