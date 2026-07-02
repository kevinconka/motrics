"""Tests for the box-matching primitives."""

import motrics
import pytest


def test_perfect_matches() -> None:
    boxes = [(0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 30.0, 30.0)]
    result = motrics.match_boxes(boxes, boxes)
    assert result.matches == [(0, 0), (1, 1)]
    assert result.scores == pytest.approx([1.0, 1.0])
    assert result.unmatched_a == []
    assert result.unmatched_b == []


def test_threshold_leaves_weak_pairs_unmatched() -> None:
    a = [(0.0, 0.0, 10.0, 10.0)]
    # Overlaps a with IoU 1/3, below the default 0.5 threshold.
    b = [(5.0, 0.0, 15.0, 10.0)]
    result = motrics.match_boxes(a, b)
    assert result.matches == []
    assert result.unmatched_a == [0]
    assert result.unmatched_b == [0]

    # Lowering the threshold recovers the match.
    result = motrics.match_boxes(a, b, iou_threshold=0.3)
    assert result.matches == [(0, 0)]


def test_unequal_set_sizes() -> None:
    a = [(0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 30.0, 30.0)]
    b = [(0.0, 0.0, 10.0, 10.0)]
    result = motrics.match_boxes(a, b)
    assert result.matches == [(0, 0)]
    assert result.unmatched_a == [1]
    assert result.unmatched_b == []


def test_hungarian_maximises_total_iou() -> None:
    # Pairing by max total IoU gives (0,1)+(1,0)=1.667, not the
    # diagonal (0,0)+(1,1)=1.5. Hungarian must never score below greedy.
    a = [(0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 6.0, 10.0)]
    b = [(0.0, 0.0, 9.0, 10.0), (0.0, 0.0, 10.0, 10.0)]

    greedy = motrics.match_boxes(a, b, iou_threshold=0.1, method="greedy")
    hungarian = motrics.match_boxes(a, b, iou_threshold=0.1, method="hungarian")

    assert hungarian.matches == [(0, 1), (1, 0)]
    assert sum(hungarian.scores) >= sum(greedy.scores)
    assert sum(hungarian.scores) == pytest.approx(1.0 + 2.0 / 3.0)


def test_empty_inputs() -> None:
    result = motrics.match_boxes([], [(0.0, 0.0, 1.0, 1.0)])
    assert result.matches == []
    assert result.unmatched_a == []
    assert result.unmatched_b == [0]


def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="unknown method"):
        motrics.match_boxes([], [], method="bogus")


def test_repr() -> None:
    boxes = [(0.0, 0.0, 10.0, 10.0)]
    assert "Matching(" in repr(motrics.match_boxes(boxes, boxes))
