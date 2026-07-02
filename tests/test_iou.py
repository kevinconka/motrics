"""Tests for the IoU primitives."""

import motrics
import pytest

UNIT = (0.0, 0.0, 10.0, 10.0)


def test_identical_boxes() -> None:
    assert motrics.iou(UNIT, UNIT) == pytest.approx(1.0)


def test_disjoint_boxes() -> None:
    assert motrics.iou(UNIT, (20.0, 20.0, 30.0, 30.0)) == pytest.approx(0.0)


def test_half_overlap() -> None:
    # inter = 5*10 = 50, union = 100 + 100 - 50 = 150.
    assert motrics.iou(UNIT, (5.0, 0.0, 15.0, 10.0)) == pytest.approx(1.0 / 3.0)


def test_degenerate_box() -> None:
    assert motrics.iou((0.0, 0.0, 0.0, 0.0), UNIT) == pytest.approx(0.0)


def test_iou_matrix_shape_and_values() -> None:
    boxes_a = [UNIT, (20.0, 20.0, 30.0, 30.0)]
    boxes_b = [UNIT]
    matrix = motrics.iou_matrix(boxes_a, boxes_b)
    assert len(matrix) == 2
    assert len(matrix[0]) == 1
    assert matrix[0][0] == pytest.approx(1.0)
    assert matrix[1][0] == pytest.approx(0.0)


def test_iou_matrix_empty() -> None:
    assert motrics.iou_matrix([], [UNIT]) == []
