"""Parity: ``motrics.iou_3d`` vs an independent shapely reference.

The tricky part of oriented-3D-box IoU is the bird's-eye-view footprint
intersection — a rotated-rectangle overlap. motrics computes it with a
hand-rolled Sutherland-Hodgman convex clip; this test checks that area against
shapely's own (independent, robust) polygon intersection, over many random
rotated boxes, and then composes the trivial height-overlap / volume terms the
same way. shapely is a reference dependency (the ``parity`` group), so the test
is skipped where it isn't installed.
"""

from __future__ import annotations

import math
import random

import motrics
import numpy as np
import pytest

shapely = pytest.importorskip("shapely")
from shapely.geometry import Polygon  # noqa: E402

Box3d = tuple[float, float, float, float, float, float, float]


def _bev_corners(b: Box3d) -> list[tuple[float, float]]:
    x, _y, z, length, width, _h, yaw = b
    c, s = math.cos(yaw), math.sin(yaw)
    hl, hw = length / 2.0, width / 2.0
    return [
        (x + dx * c + dz * s, z - dx * s + dz * c)
        for dx, dz in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))
    ]


def _ref_iou_3d(a: Box3d, b: Box3d) -> float:
    """Reference 3D IoU: shapely BEV intersection x height overlap / union."""
    bev = Polygon(_bev_corners(a)).intersection(Polygon(_bev_corners(b))).area
    top = min(a[1] + a[5] / 2.0, b[1] + b[5] / 2.0)
    bot = max(a[1] - a[5] / 2.0, b[1] - b[5] / 2.0)
    inter = bev * max(top - bot, 0.0)
    vol_a, vol_b = a[3] * a[4] * a[5], b[3] * b[4] * b[5]
    union = vol_a + vol_b - inter
    return inter / union if union > 0 else 0.0


def _random_box(rng: random.Random) -> Box3d:
    return (
        rng.uniform(-3.0, 3.0),  # x
        rng.uniform(-3.0, 3.0),  # y
        rng.uniform(-3.0, 3.0),  # z
        rng.uniform(1.0, 5.0),  # l
        rng.uniform(1.0, 5.0),  # w
        rng.uniform(1.0, 5.0),  # h
        rng.uniform(0.0, 2.0 * math.pi),  # yaw
    )


def test_iou_3d_matches_shapely() -> None:
    rng = random.Random(0)
    overlaps = 0
    for _ in range(2000):
        a, b = _random_box(rng), _random_box(rng)
        got = motrics.iou_3d(a, b)
        assert got == pytest.approx(_ref_iou_3d(a, b), abs=1e-9)
        if got > 0:
            overlaps += 1
    # The centres are close enough that many pairs genuinely overlap; otherwise
    # the test would be vacuous.
    assert overlaps > 200


def test_iou_3d_matrix_matches_pairwise() -> None:
    rng = random.Random(1)
    a = [_random_box(rng) for _ in range(5)]
    b = [_random_box(rng) for _ in range(3)]
    matrix = motrics.iou_3d_matrix(a, b)
    assert len(matrix) == 5
    assert all(len(row) == 3 for row in matrix)
    for i, ba in enumerate(a):
        for j, bb in enumerate(b):
            assert matrix[i][j] == pytest.approx(motrics.iou_3d(ba, bb), abs=1e-12)


def test_iou_3d_matrix_numpy_matches_lists() -> None:
    rng = random.Random(2)
    a = [_random_box(rng) for _ in range(4)]
    b = [_random_box(rng) for _ in range(4)]
    from_lists = motrics.iou_3d_matrix(a, b)
    from_numpy = motrics.iou_3d_matrix(
        np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    )
    assert from_numpy == from_lists


def test_iou_3d_matrix_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"shape \(N, 7\)"):
        motrics.iou_3d_matrix(
            np.zeros((2, 4), dtype=np.float64), np.zeros((2, 7), dtype=np.float64)
        )


def test_identical_and_disjoint() -> None:
    box: Box3d = (0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.7)
    assert motrics.iou_3d(box, box) == pytest.approx(1.0)
    far: Box3d = (100.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.7)
    assert motrics.iou_3d(box, far) == 0.0
