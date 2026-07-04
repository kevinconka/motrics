"""Tests for KITTI 2D-box ingest and end-to-end metric integration."""

from pathlib import Path

import motrics
import pytest

# frame, track_id, type, truncated, occluded, alpha, left, top, right, bottom,
# height, width, length, x, y, z, ry. KITTI boxes are already xyxy.
GT_TEXT = """\
1 1 Pedestrian 0 0 0 0 0 10 10 0 0 0 0 0 0 0
2 1 Pedestrian 0 0 0 0 0 10 10 0 0 0 0 0 0 0
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_load_kitti_parses_xyxy_boxes(tmp_path: Path) -> None:
    text = "1 7 Pedestrian 0 0 0 10 20 40 60 0 0 0 0 0 0 0\n"
    frames = motrics.load_kitti(_write(tmp_path, "res.txt", text))
    ids, boxes, classes = frames[1]
    assert ids == [7]
    assert boxes == [(10.0, 20.0, 40.0, 60.0)]
    assert classes == [4]  # pedestrian


def test_load_kitti_drops_negative_ids(tmp_path: Path) -> None:
    text = "1 -1 Pedestrian 0 0 0 0 0 10 10 0 0 0 0 0 0 0\n"
    frames = motrics.load_kitti(_write(tmp_path, "res.txt", text))
    assert frames == {}


def test_load_kitti_gt_separates_dontcare_regions(tmp_path: Path) -> None:
    text = (
        "1 1 Pedestrian 0 0 0 0 0 10 10 0 0 0 0 0 0 0\n"
        "1 -1 DontCare 0 0 0 20 20 30 30 0 0 0 0 0 0 0\n"
    )
    gt, ignore = motrics.load_kitti_gt(_write(tmp_path, "gt.txt", text))
    ids, boxes, classes, truncation, occlusion = gt[1]
    assert ids == [1]
    assert boxes == [(0.0, 0.0, 10.0, 10.0)]
    assert classes == [4]
    assert truncation == [0]
    assert occlusion == [0]
    assert ignore == {1: [(20.0, 20.0, 30.0, 30.0)]}


def test_load_kitti_gt_reads_truncation_and_occlusion(tmp_path: Path) -> None:
    text = "1 1 Pedestrian 1 3 0 0 0 10 10 0 0 0 0 0 0 0\n"
    gt, _ignore = motrics.load_kitti_gt(_write(tmp_path, "gt.txt", text))
    _ids, _boxes, _classes, truncation, occlusion = gt[1]
    assert truncation == [1]
    assert occlusion == [3]


def test_preprocess_kitti_rejects_unknown_class() -> None:
    with pytest.raises(ValueError, match="unknown class"):
        motrics.preprocess_kitti({}, {}, {}, "bicycle")


def test_preprocess_kitti_drops_pred_matched_to_distractor() -> None:
    gt = {
        1: (
            [1, 2],
            [(0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 10.0, 10.0)],
            [4, 5],  # pedestrian, person (distractor for pedestrian)
            [0, 0],
            [0, 0],
        )
    }
    pred = {
        1: (
            [10, 20],
            [(0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 10.0, 10.0)],
            [4, 4],
        )
    }
    gt_ids, _gt_boxes, pred_ids, _pred_boxes = motrics.preprocess_kitti(
        gt, {}, pred, "pedestrian"
    )
    assert gt_ids == [[1]]  # distractor (2) dropped
    assert pred_ids == [[10]]  # pred matched to the distractor (20) dropped


def test_preprocess_kitti_drops_occluded_and_truncated_gt() -> None:
    gt = {
        1: (
            [1, 2, 3],
            [
                (0.0, 0.0, 10.0, 10.0),
                (20.0, 20.0, 10.0, 10.0),
                (40.0, 40.0, 10.0, 10.0),
            ],
            [4, 4, 4],
            [0, 1, 0],  # gt 2 truncated (> max_truncation=0)
            [0, 0, 3],  # gt 3 occluded (> max_occlusion=2)
        )
    }
    pred = {1: ([], [], [])}
    gt_ids, _gt_boxes, _pred_ids, _pred_boxes = motrics.preprocess_kitti(
        gt, {}, pred, "pedestrian"
    )
    assert gt_ids == [[1]]


def test_preprocess_kitti_drops_short_unmatched_pred() -> None:
    pred = {1: ([10], [(0.0, 0.0, 10.0, 20.0)], [4])}  # height 20 <= min_height 25
    _gt_ids, _gt_boxes, pred_ids, _pred_boxes = motrics.preprocess_kitti(
        {}, {}, pred, "pedestrian"
    )
    assert pred_ids == [[]]


def test_preprocess_kitti_drops_pred_inside_ignore_region() -> None:
    pred = {1: ([10], [(2.0, 2.0, 18.0, 28.0)], [4])}  # height 26, inside region
    ignore = {1: [(0.0, 0.0, 20.0, 30.0)]}
    _gt_ids, _gt_boxes, pred_ids, _pred_boxes = motrics.preprocess_kitti(
        {}, ignore, pred, "pedestrian"
    )
    assert pred_ids == [[]]


def test_preprocess_kitti_keeps_real_false_positive() -> None:
    pred = {1: ([10], [(0.0, 0.0, 10.0, 30.0)], [4])}  # height 30, no gt, no ignore
    _gt_ids, _gt_boxes, pred_ids, _pred_boxes = motrics.preprocess_kitti(
        {}, {}, pred, "pedestrian"
    )
    assert pred_ids == [[10]]


def test_preprocess_kitti_excludes_other_classes() -> None:
    gt = {1: ([1], [(0.0, 0.0, 10.0, 10.0)], [1], [0], [0])}  # car, irrelevant here
    pred = {1: ([10], [(0.0, 0.0, 10.0, 10.0)], [1])}
    gt_ids, _gt_boxes, pred_ids, _pred_boxes = motrics.preprocess_kitti(
        gt, {}, pred, "pedestrian"
    )
    assert gt_ids == [[]]
    assert pred_ids == [[]]


def test_end_to_end_perfect_sequence(tmp_path: Path) -> None:
    # Identical gt and prediction -> perfect scores across every metric.
    gt, ignore = motrics.load_kitti_gt(_write(tmp_path, "gt.txt", GT_TEXT))
    pred = motrics.load_kitti(_write(tmp_path, "res.txt", GT_TEXT))
    args = motrics.preprocess_kitti(gt, ignore, pred, "pedestrian")

    clear = motrics.compute_clear(*args)
    identity = motrics.compute_identity(*args)
    hota = motrics.compute_hota(*args)

    assert clear.mota == pytest.approx(1.0)
    assert clear.num_switches == 0
    assert identity.idf1 == pytest.approx(1.0)
    assert hota.hota == pytest.approx(1.0)
