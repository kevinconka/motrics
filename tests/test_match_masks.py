"""Tests for the mask-matching primitives (`match_masks`)."""

import motrics
import numpy as np
import pytest


def _mask(row_start: int, col_start: int) -> motrics.Mask:
    """A 1x1 foreground pixel at `(row_start, col_start)` in a 4x4 grid."""
    bitmap = np.zeros((4, 4), dtype=np.uint8)
    bitmap[row_start, col_start] = 1
    return motrics.mask_encode(bitmap)


def test_perfect_matches() -> None:
    masks = [_mask(0, 0), _mask(2, 2)]
    result = motrics.match_masks(masks, masks)
    assert result.matches == [(0, 0), (1, 1)]
    assert result.scores == pytest.approx([1.0, 1.0])
    assert result.unmatched_a == []
    assert result.unmatched_b == []


def test_disjoint_masks_stay_unmatched() -> None:
    result = motrics.match_masks([_mask(0, 0)], [_mask(3, 3)])
    assert result.matches == []
    assert result.unmatched_a == [0]
    assert result.unmatched_b == [0]


def test_unequal_set_sizes() -> None:
    a = [_mask(0, 0), _mask(2, 2)]
    b = [_mask(0, 0)]
    result = motrics.match_masks(a, b)
    assert result.matches == [(0, 0)]
    assert result.unmatched_a == [1]
    assert result.unmatched_b == []


def test_greedy_method() -> None:
    result = motrics.match_masks([_mask(0, 0)], [_mask(0, 0)], method="greedy")
    assert result.matches == [(0, 0)]


def test_empty_inputs() -> None:
    result = motrics.match_masks([], [_mask(0, 0)])
    assert result.matches == []
    assert result.unmatched_a == []
    assert result.unmatched_b == [0]


def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="unknown method"):
        motrics.match_masks([], [], method="bogus")


def test_size_mismatch_raises() -> None:
    a = motrics.Mask((4, 4), [0, 16])
    b = motrics.Mask((3, 3), [0, 9])
    with pytest.raises(ValueError, match="mask size mismatch"):
        motrics.match_masks([a], [b])


def test_repr() -> None:
    masks = [_mask(0, 0)]
    assert "Matching(" in repr(motrics.match_masks(masks, masks))
