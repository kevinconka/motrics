"""Tests for KITTI-MOTS ingest and end-to-end metric integration."""

from __future__ import annotations

from pathlib import Path

import motrics
import numpy as np
import pytest

_H, _W = 4, 4


def _rle(row_start: int, row_end: int, col_start: int, col_end: int) -> str:
    """A compressed RLE string for a filled rectangle in a `_H` x `_W` grid."""
    bitmap = np.zeros((_H, _W), dtype=np.uint8)
    bitmap[row_start:row_end, col_start:col_end] = 1
    return motrics.mask_encode(bitmap).to_coco()


_MASK_A = _rle(0, 2, 0, 2)  # top-left 2x2 block
_MASK_B = _rle(2, 4, 2, 4)  # bottom-right 2x2 block, disjoint from A


def _row(frame: int, obj_id: int, class_id: int, rle: str) -> str:
    return f"{frame} {obj_id} {class_id} {_H} {_W} {rle}"


def _write(tmp_path: Path, name: str, lines: list[str]) -> Path:
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_load_kitti_mots_parses_masks(tmp_path: Path) -> None:
    text = [_row(1, 7, 2, _MASK_A)]
    frames = motrics.load_kitti_mots(_write(tmp_path, "res.txt", text))
    ids, masks, classes = frames[1]
    assert ids == [7]
    assert masks[0].to_coco() == _MASK_A
    assert classes == [2]  # pedestrian


def test_load_kitti_mots_gt_separates_ignore_regions(tmp_path: Path) -> None:
    text = [
        _row(1, 1, 2, _MASK_A),
        _row(1, 10000, 10, _MASK_B),  # ignore/DontCare region
    ]
    gt, ignore = motrics.load_kitti_mots_gt(_write(tmp_path, "gt.txt", text))
    ids, masks, classes = gt[1]
    assert ids == [1]
    assert masks[0].to_coco() == _MASK_A
    assert classes == [2]
    assert len(ignore[1]) == 1
    assert ignore[1][0].to_coco() == _MASK_B


def test_preprocess_kitti_mots_rejects_unknown_class() -> None:
    with pytest.raises(ValueError, match="unknown class"):
        motrics.preprocess_kitti_mots({}, {}, "bicycle")


def test_preprocess_kitti_mots_excludes_other_classes() -> None:
    gt = {1: ([1], [motrics.Mask.from_coco((_H, _W), _MASK_A)], [1])}  # car
    pred = {1: ([10], [motrics.Mask.from_coco((_H, _W), _MASK_A)], [1])}
    gt_ids, pred_ids, _similarity = motrics.preprocess_kitti_mots(
        gt, pred, "pedestrian"
    )
    assert gt_ids == [[]]
    assert pred_ids == [[]]


def test_preprocess_kitti_mots_keeps_all_gt_regardless_of_match() -> None:
    # Two gt pedestrians, only one matched by a prediction - both kept.
    gt = {
        1: (
            [1, 2],
            [
                motrics.Mask.from_coco((_H, _W), _MASK_A),
                motrics.Mask.from_coco((_H, _W), _MASK_B),
            ],
            [2, 2],
        )
    }
    pred = {1: ([10], [motrics.Mask.from_coco((_H, _W), _MASK_A)], [2])}
    gt_ids, pred_ids, similarity = motrics.preprocess_kitti_mots(gt, pred, "pedestrian")
    assert gt_ids == [[1, 2]]
    assert pred_ids == [[10]]
    assert similarity == [[[pytest.approx(1.0)], [pytest.approx(0.0)]]]


def test_preprocess_kitti_mots_drops_pred_inside_ignore_region() -> None:
    pred = {1: ([10], [motrics.Mask.from_coco((_H, _W), _MASK_B)], [2])}
    ignore = {1: [motrics.Mask.from_coco((_H, _W), _MASK_B)]}
    _gt_ids, pred_ids, _similarity = motrics.preprocess_kitti_mots(
        {}, pred, "pedestrian", ignore_regions=ignore
    )
    assert pred_ids == [[]]


def test_preprocess_kitti_mots_keeps_real_false_positive() -> None:
    # No gt, no ignore region -> a real, unmatched false positive is kept.
    pred = {1: ([10], [motrics.Mask.from_coco((_H, _W), _MASK_A)], [2])}
    _gt_ids, pred_ids, _similarity = motrics.preprocess_kitti_mots(
        {}, pred, "pedestrian"
    )
    assert pred_ids == [[10]]


def test_end_to_end_perfect_sequence(tmp_path: Path) -> None:
    # Identical gt and prediction across two frames -> perfect scores.
    text = [_row(1, 1, 2, _MASK_A), _row(2, 1, 2, _MASK_A)]
    gt, ignore = motrics.load_kitti_mots_gt(_write(tmp_path, "gt.txt", text))
    pred = motrics.load_kitti_mots(_write(tmp_path, "res.txt", text))
    gt_ids, pred_ids, similarity = motrics.preprocess_kitti_mots(
        gt, pred, "pedestrian", ignore_regions=ignore
    )

    clear = motrics.compute_clear_from_similarity(gt_ids, pred_ids, similarity)
    identity = motrics.compute_identity_from_similarity(gt_ids, pred_ids, similarity)
    hota = motrics.compute_hota_from_similarity(gt_ids, pred_ids, similarity)

    assert clear.mota == pytest.approx(1.0)
    assert clear.num_switches == 0
    assert identity.idf1 == pytest.approx(1.0)
    assert hota.hota == pytest.approx(1.0)
