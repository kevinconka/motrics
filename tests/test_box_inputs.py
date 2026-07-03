"""Tests for `box_format` (xyxy/xywh) and zero-copy NumPy box input.

numpy is a required runtime dependency of the core (not optional here, unlike
the `parity`-gated tests), so these run unconditionally.
"""

from __future__ import annotations

import motrics
import numpy as np
import pytest

from benchmarks.fixtures import make_synthetic

UNIT_XYXY = (0.0, 0.0, 10.0, 10.0)
UNIT_XYWH = (0.0, 0.0, 10.0, 10.0)  # same box: x=0,y=0,w=10,h=10 -> (0,0,10,10)
OFFSET_XYXY = (5.0, 5.0, 15.0, 15.0)
OFFSET_XYWH = (5.0, 5.0, 10.0, 10.0)  # x=5,y=5,w=10,h=10 -> (5,5,15,15)


def test_iou_xywh_matches_xyxy() -> None:
    xyxy = motrics.iou(UNIT_XYXY, OFFSET_XYXY)
    xywh = motrics.iou(UNIT_XYWH, OFFSET_XYWH, box_format="xywh")
    assert xywh == pytest.approx(xyxy)


def test_iou_matrix_xywh_matches_xyxy() -> None:
    xyxy = motrics.iou_matrix([UNIT_XYXY], [OFFSET_XYXY])
    xywh = motrics.iou_matrix([UNIT_XYWH], [OFFSET_XYWH], box_format="xywh")
    assert np.allclose(xywh, xyxy)


def test_match_boxes_xywh_matches_xyxy() -> None:
    xyxy = motrics.match_boxes([UNIT_XYXY], [OFFSET_XYXY], iou_threshold=0.1)
    xywh = motrics.match_boxes(
        [UNIT_XYWH], [OFFSET_XYWH], iou_threshold=0.1, box_format="xywh"
    )
    assert xywh.matches == xyxy.matches
    assert xywh.scores == pytest.approx(xyxy.scores)


def _to_xywh(box: tuple) -> tuple:
    x1, y1, x2, y2 = box
    return (x1, y1, x2 - x1, y2 - y1)


def test_compute_clear_identity_hota_xywh_matches_xyxy() -> None:
    seq = make_synthetic()[0]
    xywh_gt = [[_to_xywh(b) for b in f] for f in seq.gt_boxes]
    xywh_pred = [[_to_xywh(b) for b in f] for f in seq.pred_boxes]

    clear_xyxy = motrics.compute_clear(
        seq.gt_ids, seq.gt_boxes, seq.pred_ids, seq.pred_boxes
    )
    clear_xywh = motrics.compute_clear(
        seq.gt_ids, xywh_gt, seq.pred_ids, xywh_pred, box_format="xywh"
    )
    assert clear_xywh.mota == pytest.approx(clear_xyxy.mota)
    assert clear_xywh.motp == pytest.approx(clear_xyxy.motp)

    identity_xyxy = motrics.compute_identity(
        seq.gt_ids, seq.gt_boxes, seq.pred_ids, seq.pred_boxes
    )
    identity_xywh = motrics.compute_identity(
        seq.gt_ids, xywh_gt, seq.pred_ids, xywh_pred, box_format="xywh"
    )
    assert identity_xywh.idf1 == pytest.approx(identity_xyxy.idf1)

    hota_xyxy = motrics.compute_hota(
        seq.gt_ids, seq.gt_boxes, seq.pred_ids, seq.pred_boxes
    )
    hota_xywh = motrics.compute_hota(
        seq.gt_ids, xywh_gt, seq.pred_ids, xywh_pred, box_format="xywh"
    )
    assert hota_xywh.hota == pytest.approx(hota_xyxy.hota)


def _as_numpy_frames(frames: list[list[tuple]]) -> list[np.ndarray]:
    return [np.asarray(f, dtype=np.float64).reshape(-1, 4) for f in frames]


def test_numpy_boxes_match_list_of_tuples() -> None:
    seq = make_synthetic()[0]
    gt_np = _as_numpy_frames(seq.gt_boxes)
    pred_np = _as_numpy_frames(seq.pred_boxes)

    clear_list = motrics.compute_clear(
        seq.gt_ids, seq.gt_boxes, seq.pred_ids, seq.pred_boxes
    )
    clear_np = motrics.compute_clear(seq.gt_ids, gt_np, seq.pred_ids, pred_np)
    assert clear_np.mota == pytest.approx(clear_list.mota)
    assert clear_np.motp == pytest.approx(clear_list.motp)
    assert clear_np.num_switches == clear_list.num_switches

    identity_list = motrics.compute_identity(
        seq.gt_ids, seq.gt_boxes, seq.pred_ids, seq.pred_boxes
    )
    identity_np = motrics.compute_identity(seq.gt_ids, gt_np, seq.pred_ids, pred_np)
    assert identity_np.idf1 == pytest.approx(identity_list.idf1)

    hota_list = motrics.compute_hota(
        seq.gt_ids, seq.gt_boxes, seq.pred_ids, seq.pred_boxes
    )
    hota_np = motrics.compute_hota(seq.gt_ids, gt_np, seq.pred_ids, pred_np)
    assert hota_np.hota == pytest.approx(hota_list.hota)


def test_numpy_iou_matrix_and_match_boxes() -> None:
    boxes_a = np.array([UNIT_XYXY, OFFSET_XYXY], dtype=np.float64)
    boxes_b = np.array([UNIT_XYXY], dtype=np.float64)
    list_result = motrics.iou_matrix([UNIT_XYXY, OFFSET_XYXY], [UNIT_XYXY])
    np_result = motrics.iou_matrix(boxes_a, boxes_b)
    assert np.allclose(np_result, list_result)

    matching = motrics.match_boxes(boxes_a, boxes_b, iou_threshold=0.1)
    assert matching.matches == [(0, 0)]


def test_numpy_non_contiguous_array_still_works() -> None:
    # Fortran-ordered array is not C-contiguous -> exercises the row-copy
    # fallback path rather than the zero-copy reinterpret-cast.
    boxes = np.asfortranarray(np.array([UNIT_XYXY, OFFSET_XYXY], dtype=np.float64))
    assert not boxes.flags["C_CONTIGUOUS"]
    result = motrics.iou_matrix(boxes, boxes)
    expected = motrics.iou_matrix([UNIT_XYXY, OFFSET_XYXY], [UNIT_XYXY, OFFSET_XYXY])
    assert np.allclose(result, expected)


def test_numpy_wrong_shape_raises() -> None:
    bad = np.zeros((3, 5), dtype=np.float64)
    with pytest.raises(ValueError, match=r"\(N, 4\)"):
        motrics.iou_matrix(bad, bad)


def test_unknown_box_format_raises() -> None:
    with pytest.raises(ValueError, match="box_format"):
        motrics.iou(UNIT_XYXY, UNIT_XYXY, box_format="cxcywh")  # ty: ignore[invalid-argument-type]
